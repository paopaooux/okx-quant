"""Offline ablation of concentration, stock entry filters, and signal exits.

Run: .venv/bin/python -m scripts.analysis.entry_rule_study
No account client or live trading module is imported.

This is a TECH/live-like diagnostic, NOT the frozen profitable combination.
For controlled ablations of that original strategy use original_rule_study.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from strategies.stocks.config import Config
from strategies.stocks.market import data
from strategies.stocks.market.universe_tech import TECH
from strategies.stocks.research import events
from strategies.stocks.research.stock_categories import classify_universe


@dataclass(frozen=True)
class Policy:
    concentration: bool = False
    entry_filter: bool = False
    signal_exit: bool = False
    slots: int = 5
    max_direction: int = 3
    max_category: int = 2
    min_hours: float = 1.0
    max_hours: float = 16.0
    max_volume_ratio: float = 6.0
    max_deviation: float = 0.07
    neutral_bars: int = 3


VARIANTS = {
    "baseline": Policy(),
    "1_concentration": Policy(concentration=True),
    "2_entry_filter": Policy(entry_filter=True),
    "3_signal_exit": Policy(signal_exit=True),
    "12": Policy(concentration=True, entry_filter=True),
    "13": Policy(concentration=True, signal_exit=True),
    "23": Policy(entry_filter=True, signal_exit=True),
    "123_all": Policy(concentration=True, entry_filter=True, signal_exit=True),
}


def first_run(mask: np.ndarray, length: int) -> int | None:
    count = 0
    for i, value in enumerate(mask):
        count = count + 1 if value else 0
        if count >= length:
            return i
    return None


def path_exit(path: pd.DataFrame, side: int, width: float,
              neutral: np.ndarray, neutral_bars: int = 3) -> dict:
    """Signals use a completed bar; exit at the next bar open.

    Stops win ambiguous OHLC ties. Gap-through stops use the worse open.
    The last row is the timeout execution bar, not another holding bar.
    """
    entry = float(path.open.iloc[0])
    stop, target = entry * (1 - side * width), entry * (1 + side * width)
    active = path.iloc[:-1]
    stop_hit = active.low.to_numpy() <= stop if side > 0 else active.high.to_numpy() >= stop
    target_hit = active.high.to_numpy() >= target if side > 0 else active.low.to_numpy() <= target
    hits = np.flatnonzero(stop_hit | target_hit)
    hit = int(hits[0]) if len(hits) else None
    invalid = first_run(neutral[:-1], neutral_bars)
    invalid = invalid + 1 if invalid is not None else None
    # Intrabar fills are conservatively timestamped at the bar close.
    base_i = hit + 1 if hit is not None else len(path) - 1
    reason = "timeout"
    px = float(path.open.iloc[-1])
    if hit is not None:
        reason = "stop" if stop_hit[hit] else "target"
        if reason == "stop":
            px = min(stop, float(path.open.iloc[hit])) if side > 0 else max(stop, float(path.open.iloc[hit]))
        else:
            px = target
    base = {"exit_ts": path.index[base_i], "exit_price": px, "reason": reason}
    early = base.copy()
    if invalid is not None and (hit is None or invalid <= hit):
        early = {"exit_ts": path.index[invalid], "exit_price": float(path.open.iloc[invalid]),
                 "reason": "signal_invalid"}
    return {"entry_price": entry, "base": base, "early": early}


def filter_reason(row: dict, policy: Policy) -> str | None:
    if not policy.entry_filter or row["asset"] != "stock":
        return None
    if not policy.min_hours <= row["hours_to_open"] <= policy.max_hours:
        return "hours_to_open"
    if not np.isfinite(row["volume_ratio"]):
        return "missing_volume"
    if row["volume_ratio"] > policy.max_volume_ratio:
        return "volume_ratio"
    if abs(row["deviation"]) > policy.max_deviation:
        return "deviation"
    return None


def replay(candidates: pd.DataFrame, policy: Policy, stock_cost: float,
           crypto_cost: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every raw candidate competes anew, including after an early exit."""
    active, accepted, rejected = [], [], []
    realized = 1.0
    ordered = candidates.sort_values(["entry_ts", "strength", "symbol"], ascending=[True, False, True])
    for row in ordered.to_dict("records"):
        now = row["entry_ts"]
        closed = [x for x in active if x["exit_ts"] <= now]
        for x in closed:
            realized += x["notional"] * (x["gross"] - x["exit_cost"])
        active = [x for x in active if x["exit_ts"] > now]
        reason = filter_reason(row, policy)
        if reason is None and any(x["symbol"] == row["symbol"] for x in active):
            reason = "same_symbol"
        if reason is None and len(active) >= policy.slots:
            reason = "capacity"
        if reason is None and policy.concentration:
            if sum(x["side"] == row["side"] for x in active) >= policy.max_direction:
                reason = "direction_limit"
            elif sum(x["category"] == row["category"] for x in active) >= policy.max_category:
                reason = "category_limit"
        if reason:
            rejected.append({"entry_ts": now, "symbol": row["symbol"], "reason": reason})
            continue
        prefix = "early" if policy.signal_exit else "base"
        row.update(exit_ts=row[f"{prefix}_exit_ts"], exit_price=row[f"{prefix}_exit_price"],
                   reason=row[f"{prefix}_reason"])
        row["gross"] = row["side"] * (row["exit_price"] / row["entry_price"] - 1)
        cost = (stock_cost if row["asset"] == "stock" else crypto_cost) / 1e4
        row.update(entry_cost=cost / 2, exit_cost=cost / 2, net=row["gross"] - cost,
                   hold_hours=(row["exit_ts"] - now).total_seconds() / 3600,
                   notional=realized / policy.slots)
        realized -= row["notional"] * row["entry_cost"]
        active.append(row)
        accepted.append(row)
    return pd.DataFrame(accepted), pd.DataFrame(rejected, columns=["entry_ts", "symbol", "reason"])


def equity_curve(trades: pd.DataFrame, frames: dict, days: pd.DatetimeIndex) -> pd.Series:
    """Daily mark-to-market, including entry costs while a trade remains open."""
    values = np.ones(len(days))
    if trades.empty:
        return pd.Series(values, index=days)
    for row in trades.to_dict("records"):
        values[days >= row["entry_ts"]] -= row["notional"] * row["entry_cost"]
        values[days >= row["exit_ts"]] += row["notional"] * (row["gross"] - row["exit_cost"])
        mask = (days >= row["entry_ts"]) & (days < row["exit_ts"])
        if mask.any():
            frame = frames[row["symbol"]]
            # At a midnight grid point only the preceding completed bar is known.
            positions = frame.index.searchsorted(days[mask], side="left") - 1
            prices = frame.close.iloc[positions].to_numpy()
            values[mask] += row["notional"] * row["side"] * (prices / row["entry_price"] - 1)
    return pd.Series(values, index=days)


def metrics(trades: pd.DataFrame, curve: pd.Series) -> dict:
    net = trades.net if len(trades) else pd.Series(dtype=float)
    wins, losses = net[net > 0], net[net <= 0]
    return {"trades": len(net), "win_pct": float((net > 0).mean() * 100),
            "return_pct": float((curve.iloc[-1] / curve.iloc[0] - 1) * 100),
            "max_dd_pct": float((curve / curve.cummax() - 1).min() * 100),
            "avg_net_bps": float(net.mean() * 1e4),
            "avg_win_bps": float(wins.mean() * 1e4),
            "avg_loss_bps": float(losses.mean() * 1e4),
            "profit_factor": float(wins.sum() / -losses.sum()) if losses.sum() < 0 else None,
            "avg_hold_hours": float(trades.hold_hours.mean()) if len(net) else None,
            "signal_exits": int((trades.reason == "signal_invalid").sum()) if len(net) else 0}


def prepare(args) -> tuple[pd.DataFrame, dict, dict]:
    start, requested_end = pd.Timestamp(args.start, tz="UTC"), pd.Timestamp(args.end, tz="UTC")
    universe = classify_universe(pd.read_csv(args.stock_data / "universe.csv"))
    universe = universe[universe.ticker.isin(TECH)]
    stock = data.load_panel(sorted(universe.instId), "5m", args.stock_data)
    stock = {k: v.loc[v.index < requested_end] for k, v in stock.items()}
    crypto = {}
    oos = pd.read_csv(args.oos)
    oos["dt"] = pd.to_datetime(oos.dt, utc=True)
    for symbol in sorted(oos.symbol.unique()):
        raw = pd.read_csv(args.crypto_data / f"{symbol}.csv.gz")
        raw["ts"] = pd.to_datetime(raw.ts, unit="ms", utc=True)
        crypto[symbol] = raw.set_index("ts").sort_index().loc[lambda x: x.index < requested_end]
    # All variants use identical horizon-complete entry coverage, including early exits.
    end = min(requested_end, oos.dt.max() + pd.Timedelta(minutes=15),
              min(x.index.max() for x in crypto.values()), min(x.index.max() for x in stock.values()))
    entry_end = end - pd.Timedelta(hours=30)
    print(f"Common coverage: {start} -> {end}; last eligible entry: {entry_end}", flush=True)
    stock = {k: v.loc[v.index <= end] for k, v in stock.items()}
    cfg = Config(dislocation_bps=args.stock_trigger_bps)
    raw_events = events.off_hours_dislocation(data.to_bar_end(stock, "5m"), cfg)
    raw_events = raw_events.loc[(raw_events.event_ts >= start) & (raw_events.event_ts <= entry_end)]
    category = universe.set_index("instId").category.to_dict()
    candidates, skipped = [], {"stock_path_missing": 0, "crypto_path_missing": 0}
    policy = Policy()

    def add(symbol, stamp, side, width, asset, fields, neutral_fn, hours, step):
        frame = stock[symbol] if asset == "stock" else crypto[symbol]
        index = pd.date_range(stamp, stamp + pd.Timedelta(hours=hours), freq=step)
        path = frame.reindex(index)
        if path[["open", "high", "low", "close"]].isna().any().any():
            skipped[f"{asset}_path_missing"] += 1
            return
        result = path_exit(path, side, width, neutral_fn(path), policy.neutral_bars)
        row = dict(symbol=symbol, entry_ts=stamp, side=side, asset=asset,
                   entry_price=result["entry_price"], **fields)
        for name in ("base", "early"):
            row.update({f"{name}_{k}": v for k, v in result[name].items()})
        candidates.append(row)

    for event in raw_events.itertuples():
        # Same swap's cash-session close anchor, not an external stock quote.
        frame = stock[event.inst_id]
        pos = frame.index.get_indexer([event.event_ts])[0]
        if pos <= 0:
            skipped["stock_path_missing"] += 1
            continue
        anchor = float(frame.close.iloc[pos - 1]) / (1 + event.deviation)
        def neutral(path, anchor=anchor):
            return (path.close / anchor - 1).abs().to_numpy() <= args.stock_trigger_bps / 2e4
        add(event.inst_id, event.event_ts, event.side, .06, "stock",
            dict(category=category[event.inst_id], strength=abs(event.deviation),
                 deviation=event.deviation, volume_ratio=event.volume_ratio,
                 hours_to_open=event.hours_to_open), neutral, 30, "5min")
    print(f"Stock events: {len(raw_events)}; building crypto candidates", flush=True)
    for symbol, group in oos.groupby("symbol"):
        group = group.sort_values("dt").set_index("dt")
        scores = group.p_up
        is_flat = scores.notna() & (scores < group["hi_0.01"]) & (scores > group["lo_0.01"])
        eligible = group.loc[(group.index + pd.Timedelta(minutes=15) >= start) &
                             (group.index + pd.Timedelta(minutes=15) <= entry_end) &
                             ((scores >= group["hi_0.01"]) | (scores <= group["lo_0.01"]))]
        for stamp, row in eligible.iterrows():
            if "eligible" in row and not bool(row.eligible):
                continue
            if not np.isfinite(row.width_c) or row.width_c <= 0:
                continue
            def neutral(path, flat=is_flat):
                # OOS timestamp is the bar start; its score is known at bar end.
                return flat.reindex(path.index, fill_value=False).to_numpy(dtype=bool)
            add(symbol, stamp + pd.Timedelta(minutes=15),
                1 if row.p_up >= row["hi_0.01"] else -1, row.width_c, "crypto",
                dict(category="crypto", strength=abs(row.p_up - .5),
                     deviation=np.nan, volume_ratio=np.nan, hours_to_open=np.nan),
                neutral, 12, "15min")
    meta = {"start": str(start), "end": str(end), "entry_end": str(entry_end),
            "requested_end": str(requested_end), "raw_stock_events": len(raw_events),
            "candidates": len(candidates), "coverage_exclusions": skipped,
            "stock_trigger_bps": args.stock_trigger_bps,
            "policies": {k: asdict(v) for k, v in VARIANTS.items()},
            "costs": {"base": {"stock_round_trip_bps": 44, "crypto_round_trip_bps": 10},
                      "fee5_slip12": {"stock_round_trip_bps": 34, "crypto_round_trip_bps": 10},
                      "stress": {"stock_round_trip_bps": 68, "crypto_round_trip_bps": 16}},
            "cost_note": "fee5_slip12 uses observed 5bp single-side fees and assumed 12bp stock slippage per side; funding excluded; costs charged on entry notional.",
            "inputs": {"stock_data": str(args.stock_data), "crypto_data": str(args.crypto_data),
                       "oos": str(args.oos), "oos_sha256": hashlib.sha256(args.oos.read_bytes()).hexdigest()},
            "sizing": "20% of realized equity per entry; entry fee charged immediately; daily MTM drawdown",
            "limitations": [
                "All dates have been inspected previously; calendar splits are not fresh holdout evidence.",
                "Stock trigger defaults to live 150bp, not prior research 600bp; +/-6% and 30h exits.",
                "Crypto uses Binance OHLC and historical OOS scores, not OKX live model predictions.",
                "OHLC barriers and next-open fills approximate live polling; no live stale-signal reentries.",
                "Full uninterrupted paths are required equally for all variants; missing paths are excluded.",
                "Current TECH universe and hand-curated categories may introduce selection bias.",
                "Stocks use swap price displacement, not measured fair-value deviation from underlying stock.",
            ]}
    return pd.DataFrame(candidates), {**stock, **crypto}, meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2026-02-28")
    parser.add_argument("--end", default="2026-09-07")
    parser.add_argument("--stock-trigger-bps", type=float, default=150)
    parser.add_argument("--stock-data", type=Path, default=Path("data/stocks_swap"))
    parser.add_argument("--crypto-data", type=Path, default=Path("data/klines"))
    parser.add_argument("--oos", type=Path, default=Path("results/crypto/oos_dir_c_roll730.csv.gz"))
    parser.add_argument("--output", type=Path, default=Path("results/entry_rule_study") /
                        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    args = parser.parse_args()
    candidates, frames, meta = prepare(args)
    args.output.mkdir(parents=True, exist_ok=False)
    candidates.to_csv(args.output / "candidates.csv", index=False)
    meta["candidate_sha256"] = hashlib.sha256((args.output / "candidates.csv").read_bytes()).hexdigest()
    meta["source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    meta["generated_at"] = datetime.now(timezone.utc).isoformat()
    start, end = pd.Timestamp(meta["start"]), pd.Timestamp(meta["end"])
    days = pd.date_range(start.normalize(), end.normalize(), freq="D")
    if days[-1] < end:
        days = days.append(pd.DatetimeIndex([end]))
    summaries, monthly, segments, components = [], [], [], []
    curves = pd.DataFrame(index=days)
    for cost_name, stock_cost, crypto_cost in [("base", 44, 10), ("fee5_slip12", 34, 10), ("stress", 68, 16)]:
        for name, policy in VARIANTS.items():
            trades, rejected = replay(candidates, policy, stock_cost, crypto_cost)
            curve = equity_curve(trades, frames, days)
            key = dict(variant=name, cost=cost_name)
            summaries.append({**key, **metrics(trades, curve)})
            curves[f"{name}_{cost_name}"] = curve
            if cost_name == "base":
                trades.to_csv(args.output / f"trades_{name}.csv", index=False)
                rejected.to_csv(args.output / f"rejected_{name}.csv", index=False)
                for asset, part in trades.groupby("asset"):
                    components.append({**key, "asset": asset, "trades": len(part),
                                       "win_pct": (part.net > 0).mean() * 100,
                                       "avg_net_bps": part.net.mean() * 1e4,
                                       "pnl_pct_initial": (part.notional * part.net).sum() * 100})
            # Continuous book across boundaries; returns use MTM at midnight.
            boundaries = pd.date_range(start.normalize().replace(day=1), end, freq="MS")
            for lo in boundaries:
                hi = min(lo + pd.offsets.MonthBegin(1), end)
                lo = max(lo, start)
                if lo >= hi:
                    continue
                part = trades.loc[(trades.entry_ts >= lo) & (trades.entry_ts < hi)]
                section = curve.loc[(curve.index >= lo) & (curve.index <= hi)]
                monthly.append({**key, "month": str(lo)[:7], **metrics(part, section)})
            for label, lo, hi in [("before_august", start, pd.Timestamp("2026-08-01", tz="UTC")),
                                  ("august_onward", pd.Timestamp("2026-08-01", tz="UTC"), end)]:
                if lo >= hi or lo < start or hi > end:
                    continue
                part = trades.loc[(trades.entry_ts >= lo) & (trades.entry_ts < hi)]
                section = curve.loc[(curve.index >= lo) & (curve.index <= hi)]
                segments.append({**key, "segment": label, **metrics(part, section)})
    for name, rows in [("summary", summaries), ("monthly", monthly), ("segments", segments),
                       ("components", components)]:
        pd.DataFrame(rows).to_csv(args.output / f"{name}.csv", index=False)
    curves.to_csv(args.output / "daily_equity.csv", index_label="ts")
    (args.output / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    write_report(args.output, meta, summaries, segments)
    print(pd.DataFrame(summaries).round(3).to_string(index=False))
    print(pd.DataFrame(segments).round(3).to_string(index=False))
    print(f"Artifacts: {args.output}")


def write_report(output: Path, meta: dict, summaries: list, segments: list):
    def table(frame, columns):
        lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
        for row in frame[columns].itertuples(index=False, name=None):
            lines.append("| " + " | ".join(f"{v:.2f}" if isinstance(v, float) else str(v) for v in row) + " |")
        return "\n".join(lines)

    summary = pd.DataFrame(summaries)
    segment = pd.DataFrame(segments)
    columns = ["variant", "trades", "win_pct", "return_pct", "max_dd_pct", "avg_net_bps", "profit_factor"]
    text = [
        "# Entry Rule Study",
        f"Window: {meta['start']} to {meta['end']}. Last eligible entry: {meta['entry_end']}.",
        f"Stock trigger: {meta['stock_trigger_bps']}bp. TECH stock pool plus eight crypto symbols; shared five slots.",
        "## Rules",
        "1. At most three positions in one direction and two in one category. Crypto is one category.",
        "2. Stock events only: 1-16 hours to cash open, known volume ratio <=6, absolute displacement <=7%. Reject the first event if it fails; do not wait for a later crossing.",
        "3. Stock: displacement <=half the entry trigger for three consecutive 5m closes. Crypto: scores strictly between fold-specific tails for three consecutive 15m closes. Exit at next open. A prior stop takes precedence.",
        "Baseline: stock +/-6% barriers, 30h maximum; crypto width_c barriers, 12h maximum. No per-day entry cap. Entry at next bar open after the signal is known.",
        "## Base Cost Results",
        table(summary[summary.cost == 'base'], columns),
        "## Observed Fees, Assumed Slippage",
        "Actual fee rate 5bp per side; stock slippage still assumed 12bp per side. These are not realized account returns.",
        table(summary[summary.cost == 'fee5_slip12'], columns),
        "## Cost Stress",
        table(summary[summary.cost == 'stress'], columns),
        "## Calendar Validation",
        "The book carries positions across month boundaries. Period returns are marked to market; trade statistics group by entry date and include later exits. All months were already inspected, so these are retrospective checks, not fresh holdouts or a fitted walk-forward optimization.",
        table(segment[segment.cost == 'base'], ['variant', 'segment', 'trades', 'win_pct', 'return_pct', 'max_dd_pct']),
        "## Limits",
        *["- " + line for line in meta['limitations']],
        "- " + meta['cost_note'],
        f"- Coverage exclusions: {meta['coverage_exclusions']}.",
        "- No production source, account configuration, orders, or running processes were changed.",
        "## Artifacts",
        "[Summary](summary.csv), [monthly breakdown](monthly.csv), [components](components.csv), [equity](daily_equity.csv), [metadata](metadata.json). Trade and rejection files include all eight variants.",
    ]
    (output / "report.md").write_text("\n\n".join(text) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
