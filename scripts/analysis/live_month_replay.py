"""Replay archived predictions, not today's model. No trading client imports.

Crypto-only diagnostic portfolio; stocks do not compete for slots. Entries use
the first full 15m candle after a recorded prediction, never a past bar open.
Fees/slippage are scenarios; hypothetical funding is not assumed to be zero.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd

STEP = pd.Timedelta(minutes=15)


def load_predictions(db_path, output):
    with sqlite3.connect(f"file:{db_path.resolve()}?mode=ro", uri=True) as db:
        # First observation preserves information actually available then.
        frame = pd.read_sql_query("""WITH ranked AS (
            SELECT id,ts,symbol,bar_ts,signal_side,p_up,width,raw_json,
              ROW_NUMBER() OVER(PARTITION BY symbol,bar_ts ORDER BY ts,id) AS rank
            FROM strategy_snapshots WHERE asset_type='crypto' AND p_up IS NOT NULL
          ) SELECT * FROM ranked WHERE rank=1 ORDER BY ts,symbol""", db)
    frame["observed_at"] = pd.to_datetime(frame.ts, utc=True)
    frame["bar_at"] = pd.to_datetime(frame.bar_ts, unit="ms", utc=True)
    frame["entry_ts"] = frame.observed_at.dt.ceil("15min")
    # Same clock as entry_metadata: 48 bars after the signal bar closes.
    frame["deadline"] = frame.bar_at + STEP * 49
    frame["age_s"] = (frame.observed_at-frame.bar_at).dt.total_seconds()
    frame["missing_count"] = frame.raw_json.map(
        lambda x: len(json.loads(x)["signal"]["missing_features"])
        if "missing_features" in json.loads(x).get("signal", {}) else np.nan)
    frame.to_csv(output / "predictions.csv.gz", index=False)
    return frame


def replay_exit(candles, entry_ts, deadline, side, width, slip_bps=0):
    """Conservative stop-first ordering; trigger time is a 15m interval."""
    needed = pd.date_range(entry_ts, deadline, freq=STEP)
    if not len(needed) or not needed.isin(candles.index).all():
        return None
    slip = slip_bps / 10000
    entry = float(candles.loc[entry_ts, "open"]) * (1+side*slip)
    stop, take = entry*(1-side*width), entry*(1+side*width)
    for ts, bar in candles.loc[(candles.index >= entry_ts) & (candles.index < deadline)].iterrows():
        stop_hit = bar.low <= stop if side == 1 else bar.high >= stop
        take_hit = bar.high >= take if side == 1 else bar.low <= take
        if stop_hit or take_hit:
            if stop_hit:
                base = min(float(bar.open), stop) if side == 1 else max(float(bar.open), stop)
            else:
                # Do not claim favorable gap price improvement at a target.
                base = take
            return dict(entry_px=entry, exit_px=base*(1-side*slip),
                        exit_ts=ts+STEP, reason="stop" if stop_hit else "target",
                        ambiguous=bool(stop_hit and take_hit))
    return dict(entry_px=entry, exit_px=float(candles.loc[deadline, "open"])*(1-side*slip),
                exit_ts=deadline, reason="deadline", ambiguous=False)


def replay(predictions, markets, variant, slip):
    eligible = predictions.copy()
    if variant == "recorded":
        eligible["side"] = eligible.signal_side.map({"long": 1, "short": -1}).fillna(0)
    else:
        threshold = float(variant.split("_")[1])
        eligible["side"] = np.where(eligible.p_up >= threshold, 1, 0)
    eligible = eligible[eligible.side.ne(0)].sort_values(
        ["entry_ts", "p_up", "symbol"], ascending=[True, False, True])
    active, trades, rejected = {}, [], []
    for row in eligible.itertuples():
        active = {sym: until for sym, until in active.items() if until > row.entry_ts}
        reason = None
        if row.age_s < 900 or row.age_s > 1800:
            reason = "unclosed_or_stale_bar"
        elif row.symbol in active:
            reason = "same_instrument"
        elif len(active) >= 5:
            reason = "capacity"
        elif not np.isfinite(row.width) or row.width <= 0 or row.width >= 1:
            reason = "invalid_width"
        elif row.entry_ts >= row.deadline:
            reason = "expired"
        outcome = None if reason else replay_exit(markets[row.symbol], row.entry_ts,
                                                   row.deadline, row.side, row.width, slip)
        if not reason and outcome is None:
            reason = "incomplete_market_window"
        if reason:
            rejected.append(dict(symbol=row.symbol, observed_at=row.observed_at, reason=reason))
            continue
        ratio = outcome["exit_px"] / outcome["entry_px"]
        gross = row.side * (ratio-1)
        fee = -.0005*(1+ratio)  # 5bp taker fee per side, scenario not fee-tier inference.
        trades.append(dict(symbol=row.symbol, entry_ts=row.entry_ts, observed_at=row.observed_at,
                           p_up=row.p_up, side=row.side, width=row.width,
                           gross_return=gross, fee_return=fee, after_cost_return=gross+fee,
                           pnl_per_10_usdt=10*(gross+fee), **outcome))
        active[row.symbol] = outcome["exit_ts"]
    return pd.DataFrame(trades), pd.DataFrame(rejected)


def stats(trades):
    if trades.empty:
        return dict(trades=0, pnl_per_10_usdt=0, win_pct=None, mean_return_bps=None)
    values = trades.after_cost_return
    return dict(trades=len(trades), pnl_per_10_usdt=float(trades.pnl_per_10_usdt.sum()),
                win_pct=float(values.gt(0).mean()*100), mean_return_bps=float(values.mean()*10000),
                ambiguous_bars=int(trades.ambiguous.sum()),
                gross_per_10_usdt=float(trades.gross_return.sum()*10),
                fees_per_10_usdt=float(trades.fee_return.sum()*10))


def run(db, market_root, audit_root, out):
    out.mkdir(parents=True, exist_ok=True)
    predictions = load_predictions(db, out)
    markets, coverage = {}, []
    for sym in sorted(predictions.symbol.unique()):
        path = market_root / f"{sym}.csv.gz"
        frame = pd.read_csv(path)
        frame["ts"] = pd.to_datetime(frame.ts, unit="ms", utc=True)
        frame = frame.drop_duplicates("ts").set_index("ts").sort_index()
        markets[sym] = frame
        frame.to_csv(out / f"market_{sym}.csv.gz")
        wanted = pd.date_range(predictions.entry_ts.min(),
                               min(predictions.entry_ts.max(), frame.index.max()), freq=STEP)
        coverage.append(dict(symbol=sym, first=str(frame.index.min()), last=str(frame.index.max()),
                             missing_bars=len(wanted.difference(frame.index))))
    pd.DataFrame(coverage).to_csv(out / "market_coverage.csv", index=False)
    summary, contributions = [], []
    split = pd.Timestamp("2026-09-16T00:00:00Z")
    for variant in ["recorded", "long_0.58", "long_0.57"]:
        for slip in [0, 10]:
            trades, rejected = replay(predictions, markets, variant, slip)
            trades.to_csv(out / f"trades_{variant}_{slip}.csv", index=False)
            rejected.to_csv(out / f"rejected_{variant}_{slip}.csv", index=False)
            for period in ["all", "before_sep16", "since_sep16"]:
                subset = trades if period == "all" or trades.empty else trades[
                    trades.entry_ts.lt(split) if period == "before_sep16" else trades.entry_ts.ge(split)]
                summary.append(dict(variant=variant, slip_bps_per_side=slip, period=period, **stats(subset)))
            if len(trades):
                for sym, group in trades.groupby("symbol"):
                    contributions.append(dict(variant=variant, slip_bps_per_side=slip, symbol=sym, **stats(group)))
    summary = pd.DataFrame(summary)
    summary.to_csv(out / "replay_summary.csv", index=False)
    pd.DataFrame(contributions).to_csv(out / "symbol_contributions.csv", index=False)

    # Overlapping forward returns are descriptive observations, not independent tests.
    diagnostics = []
    for row in predictions.itertuples():
        market = markets[row.symbol]
        end = row.entry_ts + STEP*48
        if row.entry_ts not in market.index or end not in market.index or not 900 <= row.age_s <= 1800:
            continue
        change = float(market.loc[end, "open"] / market.loc[row.entry_ts, "open"] - 1)
        diagnostics.append(dict(symbol=row.symbol, observed_at=row.observed_at, p_up=row.p_up,
                                missing_count=row.missing_count, forward_12h_bps=change*10000))
    diag = pd.DataFrame(diagnostics)
    diag["probability_bin"] = pd.cut(diag.p_up, [0, .45, .5, .55, .57, .58, .5890769453, 1])
    buckets = diag.groupby("probability_bin", observed=True).agg(
        observations=("p_up", "size"), mean_forward_bps=("forward_12h_bps", "mean"),
        positive_pct=("forward_12h_bps", lambda s: s.gt(0).mean()*100))
    buckets.to_csv(out / "prediction_diagnostics.csv")
    daily = predictions.groupby(predictions.observed_at.dt.strftime("%Y-%m-%d")).agg(
        observations=("p_up", "size"), longs=("signal_side", lambda s: s.eq("long").sum()),
        shorts=("signal_side", lambda s: s.eq("short").sum()), p_min=("p_up", "min"),
        p_max=("p_up", "max"), missing_min=("missing_count", "min"), missing_max=("missing_count", "max"))
    daily.to_csv(out / "daily_signal_coverage.csv")

    actual = pd.read_csv(audit_root / "closed_positions_enriched.csv")
    actual["cTime"] = pd.to_datetime(actual.cTime, utc=True)
    actual["uTime"] = pd.to_datetime(actual.uTime, utc=True)
    actual["entry_period"] = np.select(
        [actual.cTime.lt(pd.Timestamp("2026-09-07T07:56:41Z")),
         actual.cTime.lt(pd.Timestamp("2026-09-22T08:32:44Z"))],
        ["legacy", "original_policy"], default="execution_fixed")
    actual.groupby(["asset", "entry_period"], observed=True).agg(
        trades=("realizedPnl", "size"), gross=("pnl", "sum"), fees=("fee", "sum"),
        funding=("fundingFee", "sum"), net=("realizedPnl", "sum")).to_csv(out / "actual_by_regime.csv")
    actual.groupby(["asset", "instId"]).agg(trades=("realizedPnl", "size"),
        net=("realizedPnl", "sum")).to_csv(out / "actual_by_symbol.csv")
    actual.to_csv(out / "actual_trades.csv", index=False)
    manifest = dict(first_observation=str(predictions.observed_at.min()),
        last_observation=str(predictions.observed_at.max()), distinct_symbol_bars=len(predictions),
        actual_closed_positions=len(actual), actual_source=str(audit_root),
        database_read_only=True, strategy_modified=False, model_retrained=False,
        fixed_notional_per_trade_usdt=10, fee_bps_per_side=5,
        hypothetical_funding="Not included: after-cost replay results are BEFORE funding.",
        caveats=["First recorded prediction per symbol/bar; later revisions are not reconstructed.",
                 "No predictions fabricated during missing logging periods.",
                 "Next full 15m candle entry adds up to 15m; not a reconstruction of actual fills.",
                 "Crypto-only 5-slot diagnostic; no stock competition, lot rounding or compounding.",
                 "Same-candle stop and target resolved stop-first; exits timed to candle end.",
                 "Incomplete full holding windows excluded and recorded, including potential early exits.",
                 "Relaxed thresholds are exploratory; chronological halves are NOT untouched holdouts.",
                 "Missing-feature metadata absent early does not imply complete model inputs.",
                 "Forward-return observations overlap and are not independent sample counts.",
                 "No Sharpe or portfolio drawdown inferred from fixed-notional completed-trade sums."])
    manifest["prediction_sha256"] = hashlib.sha256((out / "predictions.csv.gz").read_bytes()).hexdigest()
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print("\nProbability diagnostics (overlapping windows):\n", buckets)
    print("\nCoverage:\n", pd.DataFrame(coverage).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("data/okx_demo.sqlite3"))
    parser.add_argument("--market", type=Path, default=Path("data/okx_klines"))
    parser.add_argument("--audit", type=Path, default=Path("results/live_audit_20260928"))
    parser.add_argument("--output", type=Path, default=Path("results/live_month_replay_20260928"))
    args = parser.parse_args()
    run(args.db, args.market, args.audit, args.output)
