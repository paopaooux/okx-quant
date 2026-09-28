"""Regenerate stock candidates on recovered OHLC and replay frozen allocation."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analysis import original_rule_study as original
from scripts.analysis.entry_rule_study import Policy
from scripts.analysis.us_only_study import load_candidates, stats
from scripts.combinations.run import _pooled_curve
from strategies.stocks.config import Config
from strategies.stocks.market import data, reference
from strategies.stocks.research.backtest import Rules, run
from strategies.stocks.research.events import off_hours_dislocation

SOURCE = Path("data/research_stock_history_20260928")
OUT = Path("results/reference_calendar_full_history_20260928")


def independent_candidates(events, frames):
    rules = Rules(horizon="to_open", resolve_offset_minutes=60, stop_loss_bps=300,
                  max_concurrent=100000, max_per_day=100000,
                  rank_column="abs_deviation", min_trailing_volume=0)
    # Score each candidate independently; rejected shadow trades must not
    # suppress the next event before portfolio admission is rerun.
    parts = [run(events.iloc[[i]], {events.iloc[i].inst_id: frames[events.iloc[i].inst_id]},
                 Config(), rules).trades for i in range(len(events))]
    trades = pd.concat(parts, ignore_index=True)
    if len(trades) != len(events) or trades.reason.eq("no_data").any():
        raise ValueError("Incomplete candidate pricing")
    return trades.assign(symbol=trades.inst_id, asset="stock", strategy="xstock_hybrid",
                         signal_strength=trades.signal.abs(), original_exit_ts=trades.exit_ts)


def remove_gap_affected(events, candidates, frames):
    joined = candidates.merge(events[["inst_id", "event_ts", "anchor_close_ts"]],
                              left_on=["inst_id", "entry_ts"], right_on=["inst_id", "event_ts"],
                              validate="one_to_one")
    affected = []
    for row in joined.itertuples():
        expected = pd.date_range(row.anchor_close_ts, row.exit_ts, freq="5min")
        affected.append(len(expected.difference(frames[row.inst_id].index)) > 0)
    rejected = joined.loc[affected].copy()
    return candidates.loc[~np.asarray(affected)].copy(), rejected


def audit_legacy(candidates):
    archive = load_candidates("stock")
    keys = ["symbol", "entry_ts"]
    compare = archive[keys + ["exit_ts", "gross", "entry_price"]].merge(
        candidates[keys + ["exit_ts", "gross", "entry_price"]], on=keys, how="outer",
        suffixes=("_archive", "_fresh"), indicator=True, validate="one_to_one")
    compare.to_csv(OUT / "legacy_candidate_audit.csv", index=False)
    both = compare[compare._merge.eq("both")]
    return dict(archived=len(archive), regenerated=len(candidates), matched=len(both),
                archive_only=int(compare._merge.eq("left_only").sum()),
                fresh_only=int(compare._merge.eq("right_only").sum()),
                changed_gross=int((~np.isclose(both.gross_archive, both.gross_fresh)).sum()),
                changed_entry_price=int((~np.isclose(both.entry_price_archive, both.entry_price_fresh)).sum()))


def audit_candles(universe):
    rows = []
    snapshot = Path("results/live_audit_20260911/market/5m")
    for inst in universe.instId:
        path = SOURCE / "5m" / f"{inst}.csv"
        f = pd.read_csv(path, parse_dates=["ts"]).set_index("ts")
        if f.index.has_duplicates or not f.index.is_monotonic_increasing:
            raise ValueError(f"Unsorted/duplicate candles: {inst}")
        invalid = (f[["open", "high", "low", "close"]].le(0).any(axis=1) |
                   f.high.lt(f[["open", "close", "low"]].max(axis=1)) |
                   f.low.gt(f[["open", "close"]].min(axis=1)) |
                   f.isna().any(axis=1))
        expected = pd.date_range(f.index.min(), pd.Timestamp("2026-09-04T23:55Z"), freq="5min")
        gaps = expected.difference(f.index)
        archived = pd.read_csv(snapshot / f"{inst}.csv", parse_dates=["ts"]).set_index("ts")
        overlap = f.index.intersection(archived.index)
        cols = ["open", "high", "low", "close", "volume_quote"]
        changed = int((~np.isclose(f.loc[overlap, cols], archived.loc[overlap, cols])).any(axis=1).sum())
        rows.append(dict(inst=inst, first=str(f.index.min()), last=str(f.index.max()), rows=len(f),
                         internal_or_tail_gaps=len(gaps), invalid=int(invalid.sum()),
                         snapshot_overlap=len(overlap), changed_snapshot_rows=changed,
                         sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "candle_audit.csv", index=False)
    if result.invalid.sum():
        raise ValueError("Candle audit failed; inspect candle_audit.csv")
    return dict(rows=int(result.rows.sum()), overlap=int(result.snapshot_overlap.sum()),
                changed_overlap=int(result.changed_snapshot_rows.sum()),
                internal_or_tail_gaps=int(result.internal_or_tail_gaps.sum()))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((SOURCE / "download_manifest.json").read_text())
    if manifest["errors"] or len(manifest["completed"]) != 167:
        raise ValueError("Downloads incomplete")
    universe = pd.read_csv(SOURCE / "universe.csv")
    candle_audit = audit_candles(universe)
    print("Candle audit:", candle_audit, flush=True)
    frames = data.to_bar_end(data.load_panel(sorted(universe.instId), "5m", SOURCE), "5m")
    assert len(frames) == 167
    eligible = {k: v for k, v in frames.items() if reference.reference_market(k)}
    config = Config(dislocation_bps=600)
    policy = Policy()
    crypto = original.sleeve_select(load_candidates("crypto"), policy)
    crypto["net"] = crypto.gross-.001
    _, crypto, _ = _pooled_curve(crypto, original.DAYS, 5)
    baseline_stocks = original.sleeve_select(load_candidates("stock"), policy)
    baseline = pd.concat([baseline_stocks, crypto], ignore_index=True)
    baseline["net"] = baseline.gross-np.where(baseline.asset.eq("stock"), 44, 10)/10000
    curve, accepted, _ = original.pooled(baseline, policy)
    assert len(accepted) == 299 and np.isclose((curve.iloc[-1]-1)*100, 50.277153751470195)
    print("Archived original baseline reproduced: 299 trades, +50.277154%", flush=True)
    summaries, components, months = [], [], []
    audit = None
    for variant in ("legacy_all_us_clock", "eligible_us_clock", "reference_calendars"):
        pool = frames if variant == "legacy_all_us_clock" else eligible
        events = (reference.stock_events(pool, config) if variant == "reference_calendars"
                  else off_hours_dislocation(pool, config))
        events = events[events.event_ts.le(original.END)].copy()
        events.to_csv(OUT / f"events_{variant}.csv", index=False)
        candidates = independent_candidates(events, frames)
        candidates, gap_rejected = remove_gap_affected(events, candidates, frames)
        gap_rejected.to_csv(OUT / f"gap_rejected_{variant}.csv", index=False)
        print(f"{variant}: excluded {len(gap_rejected)} candidates with gaps between anchor and exit", flush=True)
        candidates.to_csv(OUT / f"candidates_{variant}.csv", index=False)
        if variant == "legacy_all_us_clock":
            audit = audit_legacy(candidates)
            print("Fresh versus archived candidates:", audit, flush=True)
        stocks = original.sleeve_select(candidates, policy)
        for cost in (44, 68):
            for scope in ("stock_only", "combined"):
                selected = stocks.copy() if scope == "stock_only" else pd.concat([stocks, crypto], ignore_index=True)
                selected["net"] = selected.gross-np.where(selected.asset.eq("stock"), cost,
                                                         16 if cost == 68 else 10)/10000
                curve, trades, rejected = original.pooled(selected, policy)
                trades["market"] = trades.symbol.map(reference.reference_market).fillna("unmapped_or_crypto")
                summaries.append(dict(variant=variant, scope=scope, stock_cost_bps=cost, **stats(curve, trades)))
                tag = f"{variant}_{scope}_{cost}"
                trades.to_csv(OUT / f"trades_{tag}.csv", index=False)
                curve.to_csv(OUT / f"equity_{tag}.csv")
                rejected.to_csv(OUT / f"rejected_{tag}.csv", index=False)
                for (asset, market), group in trades.groupby(["asset", "market"]):
                    components.append(dict(variant=variant, scope=scope, cost=cost, asset=asset, market=market,
                        trades=len(group), contribution_pct=(group.notional*group.net).sum()*100))
                for month, group in trades.groupby(trades.entry_ts.dt.strftime("%Y-%m")):
                    months.append(dict(variant=variant, scope=scope, cost=cost, month=month, trades=len(group),
                        contribution_pct=(group.notional*group.net).sum()*100))
        print(f"Completed {variant}: {len(events)} events, {len(stocks)} admitted stock candidates", flush=True)
    pd.DataFrame(summaries).to_csv(OUT / "summary.csv", index=False)
    pd.DataFrame(components).to_csv(OUT / "components.csv", index=False)
    pd.DataFrame(months).to_csv(OUT / "monthly.csv", index=False)
    (OUT / "manifest.json").write_text(json.dumps(dict(start=str(original.START), end=str(original.END),
        mapping_version=reference.VERSION, source=str(SOURCE), legacy_audit=audit, candle_audit=candle_audit,
        download_manifest_sha256=hashlib.sha256((SOURCE / "download_manifest.json").read_bytes()).hexdigest(),
        crypto_candidates_sha256=hashlib.sha256(Path("results/original_rule_study_20260907/crypto_candidates.csv").read_bytes()).hexdigest(),
        archived_baseline_reproduced=True,
        caveats=["Current contract universe and market mapping; delisted instruments not reconstructed.",
                 "Stocks repriced from recovered OHLC; crypto uses frozen original candidate outcomes.",
                 "Original two-stage allocation, not the current unified live allocator.",
                 "Horizon-complete original study convention: entry >= start and exit <= end.",
                 "No funding, live delays, lot rounding or market impact; exact stop barrier fills.",
                 "Unconfirmed candles omitted; candidates with missing anchor-to-exit bars excluded retrospectively. Missing windows with no detected signal remain unknowable.",
                 "Daily realized equity drawdown excludes intraday/open-position risk.",
                 "Retrospective in-sample comparison; no model/threshold retuning."]), indent=2))
    print(pd.DataFrame(summaries).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
