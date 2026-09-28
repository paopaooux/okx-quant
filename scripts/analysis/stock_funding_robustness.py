"""Actual rate histories, approximate settlement notionals, paired coverage."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analysis.stock_robustness_study import (
    OUT as STUDY, STEP, VARIANTS, load_frames, marked_curve, metrics, replay,
)

OUT = Path("results/stock_funding_robustness_20260928")
SOURCE = Path("data/research_stock_funding_20260928")
PERIODS = {
    "july_september": (pd.Timestamp("2026-07-01T00:00Z"), pd.Timestamp("2026-09-26T00:00Z")),
    "july": (pd.Timestamp("2026-07-01T00:00Z"), pd.Timestamp("2026-08-01T00:00Z")),
    "august": (pd.Timestamp("2026-08-01T00:00Z"), pd.Timestamp("2026-09-01T00:00Z")),
    "september": (pd.Timestamp("2026-09-01T00:00Z"), pd.Timestamp("2026-09-26T00:00Z")),
    "recent": (pd.Timestamp("2026-09-11T00:00Z"), pd.Timestamp("2026-09-26T00:00Z")),
}


def load_candidates(path):
    frame = pd.read_csv(path)
    for col in ("event_ts", "resolve_ts", "anchor_close_ts", "entry_ts", "exit_ts",
                "exit_lower_ts", "deadline", "coverage_deadline"):
        frame[col] = pd.to_datetime(frame[col], utc=True)
    return frame


def funding_return(row, rates, frame):
    # Coverage is checked against a common preknown horizon for paired cohorts.
    if rates.index.min() > row.entry_ts or rates.index.max() < row.coverage_deadline:
        return None
    checks = rates.loc[row.entry_ts-pd.Timedelta(hours=8):row.coverage_deadline+pd.Timedelta(hours=8)]
    if checks.index.to_series().diff().gt(pd.Timedelta(hours=8)).any():
        return None
    due = rates[(rates.index > row.entry_ts) & (rates.index <= row.exit_ts)]
    cash, optimistic, ambiguous = [], 0., 0
    for stamp, rate in due.items():
        index = (frame.index+STEP).searchsorted(stamp, side="right")-1
        if index < 0 or stamp-(frame.index[index]+STEP) > STEP:
            return None
        # Historical mark price is unavailable: use the last completed perp close.
        amount = -row.side*float(rate)*float(frame.close.iloc[index])/row.entry_price
        if stamp >= row.exit_lower_ts:
            ambiguous += 1
            optimistic += max(0., amount)
            amount = min(0., amount)
        else:
            optimistic += amount
        cash.append((stamp.isoformat(), amount))
    return dict(funding=sum(x[1] for x in cash), funding_events=cash,
                funding_optimistic=optimistic, ambiguous_settlements=ambiguous)


def attach_funding(candidates, rates, frames):
    accepted, excluded = [], []
    for row in candidates.itertuples(index=False):
        found = funding_return(row, rates[row.symbol], frames[row.symbol])
        if found is None:
            excluded.append(dict(symbol=row.symbol, entry_ts=row.entry_ts, reason="funding_or_mark_coverage"))
        else:
            accepted.append(row._asdict() | found)
    return pd.DataFrame(accepted), pd.DataFrame(excluded)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frames, _, _, _ = load_frames()
    rates, hashes = {}, {}
    manifest = json.loads((SOURCE / "manifest.json").read_text())
    if manifest["errors"] or len(manifest["completed"]) != len(frames):
        raise ValueError("Funding download incomplete")
    expected_hashes = {r["inst"]: r["sha256"] for r in manifest["completed"]}
    for inst in frames:
        path = SOURCE / f"{inst}.json"
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        if hashes[str(path)] != expected_hashes[inst]:
            raise ValueError(f"Funding input changed: {inst}")
        raw = json.loads(path.read_text())
        stamps = pd.to_datetime([int(r["fundingTime"]) for r in raw], unit="ms", utc=True)
        series = pd.Series([float(r["realizedRate"]) for r in raw], index=stamps).sort_index()
        if series.index.has_duplicates or not np.isfinite(series).all():
            raise ValueError(f"Invalid funding data: {inst}")
        rates[inst] = series
    rows, coverage, components = [], [], []
    for variant in VARIANTS:
        path = STUDY / f"candidates_{variant.name}.csv"
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        candidates = load_candidates(path)
        candidates = candidates[candidates.entry_ts.ge(PERIODS["july_september"][0])]
        funded, excluded = attach_funding(candidates, rates, frames)
        excluded.to_csv(OUT / f"coverage_excluded_{variant.name}.csv", index=False)
        coverage.append(dict(variant=variant.name, candidates=len(candidates), covered=len(funded), excluded=len(excluded)))
        saved = funded.copy()
        saved["funding_events"] = saved.funding_events.map(json.dumps)
        saved.to_csv(OUT / f"candidates_{variant.name}.csv", index=False)
        unfunded = funded.copy()
        unfunded["funding"] = 0.
        unfunded["funding_events"] = [[] for _ in range(len(unfunded))]
        for period, (start, end) in PERIODS.items():
            for cost in (44, 68):
                for scenario, pool in (("unfunded_paired", unfunded), ("funded_conservative", funded)):
                    trades, _ = replay(pool, variant, cost, start, end)
                    eq = marked_curve(trades, frames, start, end)
                    rows.append(dict(variant=variant.name, period=period, cost=cost, scenario=scenario,
                        funding_contribution_pct=float((trades.notional*trades.funding).sum()*100),
                        mean_funding_bps=float(trades.funding.mean()*10000),
                        ambiguous_settlements=int(trades.ambiguous_settlements.sum()), **metrics(trades, eq)))
                    tag = f"{variant.name}_{period}_{cost}_{scenario}"
                    trades.to_csv(OUT / f"trades_{tag}.csv", index=False)
                    eq.resample("1D").last().to_csv(OUT / f"daily_{tag}.csv")
                    if scenario == "funded_conservative" and period == "july_september":
                        for inst, g in trades.groupby("symbol"):
                            components.append(dict(variant=variant.name, cost=cost, symbol=inst,
                                trades=len(g), contribution_pct=(g.notional*g.net).sum()*100,
                                funding_contribution_pct=(g.notional*g.funding).sum()*100))
        print(f"Funding replay {variant.name}: {len(funded)}/{len(candidates)} candidates covered", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "summary.csv", index=False)
    pd.DataFrame(coverage).to_csv(OUT / "coverage.csv", index=False)
    pd.DataFrame(components).to_csv(OUT / "components.csv", index=False)
    (OUT / "manifest.json").write_text(json.dumps(dict(rate_source=str(SOURCE),
        input_hashes=hashes,
        rate_manifest_sha256=hashlib.sha256((SOURCE / "manifest.json").read_bytes()).hexdigest(),
        method="Actual realized rates, last completed perpetual close as mark-notional proxy; matched coverage",
        timing="(entry, exit]; at ambiguous stop/target/deadline boundary include debits, exclude credits",
        sizing="20% realized-equity sizing; funding booked at exit for sizing, at settlement for marked curve",
        caveats=["Not exact historical mark-price cash flows, and not user account bills.",
                 "No substitution for rates before available coverage.",
                 "All variant outcomes available in history; funding never enters signal filters or admission ranking.",
                 "Funding boundary assumptions bracket uncertain bar-level exit timing; no exact intrabar timestamps.",
                 "All other OHLC execution and retrospective-sample caveats of the parent study apply."]), indent=2))
    result = pd.DataFrame(rows)
    print(result[(result.period == "july_september") & (result.cost == 68) &
                 (result.scenario == "funded_conservative")].to_string(index=False))


if __name__ == "__main__":
    main()
