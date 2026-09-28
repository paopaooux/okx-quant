"""Paired calendar correction study on frozen recent 5m candles; no orders."""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from scripts.analysis.original_rule_study import pooled
from scripts.analysis.entry_rule_study import Policy
from scripts.analysis.us_only_study import stats
from strategies.stocks.config import Config
from strategies.stocks.market import data, reference
from strategies.stocks.research.events import off_hours_dislocation
from strategies.stocks.research.backtest import Rules, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-data-dir", type=Path)
    parser.add_argument("--out", type=Path, default=Path("results/reference_calendar_study_20260928"))
    parser.add_argument("--start", default="2026-09-11T00:00Z")
    parser.add_argument("--end", default="2026-09-26T00:00Z")
    parser.add_argument("--split", default="2026-09-19T00:00Z")
    args = parser.parse_args()
    source = args.raw_data_dir or Path("results/us_only_study_20260928")
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    start, end, split = map(pd.Timestamp, (args.start, args.end, args.split))
    if not start < split < end or any(t.tzinfo is None for t in (start, split, end)):
        raise ValueError("Require timezone-aware start < split < end")
    frames, hashes = {}, {}
    paths = source.glob("5m/*.csv") if args.raw_data_dir else source.glob("recent_market_*.csv.gz")
    for path in sorted(paths):
        frame = pd.read_csv(path)
        frame["ts"] = pd.to_datetime(frame.ts, utc=True)
        inst = path.stem if args.raw_data_dir else path.name.removeprefix("recent_market_").removesuffix(".csv.gz")
        # The source snapshot is already close-time indexed, not bar-start.
        frames[inst] = frame.set_index("ts").sort_index()
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    if args.raw_data_dir:
        frames = data.to_bar_end(frames, "5m")
    if not frames or (args.raw_data_dir and (
            min(f.index.min() for f in frames.values()) >= start or
            any(f.index.max() < end + pd.Timedelta(hours=30) for f in frames.values()))):
        raise ValueError("Incomplete raw coverage: need pre-entry history and 30h exit tail")
    pd.DataFrame([dict(inst=inst, first=f.index.min(), last=f.index.max(), rows=len(f))
                  for inst, f in frames.items()]).to_csv(out / "coverage.csv", index=False)
    assert len(frames) == 167
    eligible = {inst: f for inst, f in frames.items() if reference.reference_market(inst)}
    pd.DataFrame([dict(inst=inst, reference_market=reference.reference_market(inst),
                       enabled=inst in eligible) for inst in frames]).to_csv(out / "routing.csv", index=False)
    config = Config(dislocation_bps=600)
    variants = {
        "legacy_all_us_clock": off_hours_dislocation(frames, config),
        "eligible_us_clock": off_hours_dislocation(eligible, config),
        "reference_calendars": reference.stock_events(eligible, config),
    }
    rules = Rules(horizon="to_open", resolve_offset_minutes=60, stop_loss_bps=300,
                  max_concurrent=3, max_per_day=2, rank_column="abs_deviation", min_trailing_volume=0)
    rows, components, changes = [], [], []
    for variant, candidates in variants.items():
        candidates = candidates[candidates.event_ts.ge(start) & candidates.event_ts.lt(end)].copy()
        candidates["reference_market"] = candidates.inst_id.map(reference.reference_market).fillna("disabled")
        candidates.to_csv(out / f"events_{variant}.csv", index=False)
        for period, lo, hi in [("all", start, end),
                               ("early", start, split),
                               ("late", split, end)]:
            # Fresh allocation per entry cohort, not a claim of held-out validation.
            subset = candidates[candidates.event_ts.ge(lo) & candidates.event_ts.lt(hi)]
            selected = run(subset, frames, config, rules).trades
            if selected.empty:
                raise ValueError(f"Empty comparison: {variant}/{period}")
            selected = selected.assign(symbol=selected.inst_id, asset="stock", strategy="xstock_hybrid",
                                       signal_strength=selected.signal.abs())
            for cost in [10, 30, 44, 68]:
                selected["net"] = selected.gross-cost/10000
                equity_end = max(end.normalize(), selected.exit_ts.max().normalize())
                days = pd.date_range(lo.normalize(), equity_end, freq="D")
                curve, trades, _ = pooled(selected, Policy(), days=days)
                trades["reference_market"] = trades.symbol.map(reference.reference_market).fillna("disabled")
                rows.append(dict(variant=variant, period=period, cost_bps=cost, **stats(curve, trades)))
                trades.to_csv(out / f"trades_{variant}_{period}_{cost}.csv", index=False)
                if period == "all":
                    for market, group in trades.groupby("reference_market"):
                        components.append(dict(variant=variant, cost_bps=cost, market=market,
                            trades=len(group), contribution_pct=(group.notional*group.net).sum()*100))
                    if cost == 44:
                        changes.extend(dict(variant=variant, inst=r.inst_id, entry=str(r.entry_ts), exit=str(r.exit_ts),
                                            reason=r.reason, gross=r.gross) for r in trades.itertuples())
        print(f"Completed {variant}: {len(candidates)} candidate events", flush=True)
    result = pd.DataFrame(rows)
    baseline = result[(result.variant == "legacy_all_us_clock") & (result.period == "all") & (result.cost_bps == 44)].iloc[0]
    reproduced = None
    if not args.raw_data_dir and args.start == "2026-09-11T00:00Z" and args.end == "2026-09-26T00:00Z":
        assert baseline.trades == 25 and abs(baseline.return_pct-(-6.8981347)) < .0001
        reproduced = True
    result.to_csv(out / "summary.csv", index=False)
    pd.DataFrame(components).to_csv(out / "components.csv", index=False)
    pd.DataFrame(changes).to_csv(out / "admission_comparison.csv", index=False)
    manifest = dict(start=str(start), end=str(end), source=str(source), sha256=hashes,
        mapping_version=reference.VERSION, calendar_library="exchange-calendars 4.13.2",
        enabled=len(eligible), excluded=sorted(set(frames)-set(eligible)), baseline_reproduced=reproduced,
        split=str(split), input_bar_timestamp="open" if args.raw_data_dir else "close",
        caveats=["Stock-only rerun; no crypto occupancy, live delay, funding or lot rounding.",
                 "Bar-close entries and exact barrier stops retained from original engine; gap fills can be worse.",
                 "20% per position, up to 3 concurrent stock trades and 2 entries per UTC day.",
                 "Unknown-market exclusion separated from calendar correction by eligible_us_clock control.",
                 "Current reference metadata, not reconstructed historical listing-transition dates.",
                 "Some instruments start later than the entry window; no synthetic prehistory is filled.",
                 "Short retrospective sample; early/late cohorts are exploratory, not untouched holdouts.",
                 "Older archived trade outcomes cannot reconstruct alternative reference-market signals without old OHLC.",
                 "Maximum drawdown is realized daily equity, not intraday marked equity."])
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(result[result.cost_bps.isin([10, 44, 68])].to_string(index=False))


if __name__ == "__main__":
    main()
