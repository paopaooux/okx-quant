"""Reproducible research study for the stock-perpetual off-hours gap rule.

This is intentionally separate from the live strategy entry point.  It only
loads local OKX candles, builds events, and writes train/test diagnostics.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from strategies.trade_metrics import METRIC_NOTE
from strategies.stocks.config import Config
from strategies.stocks.market import data
from strategies.stocks.market.universe_tech import TECH
from strategies.stocks.research import backtest, events, portfolio
from strategies.stocks.research.stock_categories import classify_universe, validate_categories


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=os.environ.get("STOCK_ALPHA_DATA_DIR", "data/stocks_swap"))
    parser.add_argument("--result-dir", default=os.environ.get("STOCK_ALPHA_RESULT_DIR", "results/stocks_offhours_research"))
    parser.add_argument("--pool", choices=["all", "tech"], default="all")
    parser.add_argument("--category", default="all", help="category to isolate; all reports every category")
    parser.add_argument("--test-start", default="2026-08-01")
    parser.add_argument("--dislocation-bps", type=float, default=600.0)
    parser.add_argument("--resolve-offset-minutes", type=float, default=60.0)
    parser.add_argument("--volume-ratio", type=float, default=0.0)
    parser.add_argument("--fee-bps", type=float, default=10.0)
    parser.add_argument("--slippage-bps", type=float, default=12.0)
    parser.add_argument("--direction", choices=["both", "long", "short"], default="both")
    parser.add_argument("--slots", type=int, default=4)
    parser.add_argument("--max-per-day", type=int, default=4)
    parser.add_argument("--stop-loss-bps", type=float, default=600.0)
    return parser.parse_args()


def main() -> int:
    args = _args()
    data_dir = Path(args.data_dir)
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    test_start = pd.Timestamp(args.test_start, tz="UTC")

    universe = classify_universe(pd.read_csv(data_dir / "universe.csv"))
    validate_categories(universe)
    ids = sorted(universe.instId.dropna().astype(str).unique())
    frames = data.to_bar_end(data.load_panel(ids, "5m", data_dir), "5m")
    cfg = Config(
        data_dir=str(data_dir),
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
        dislocation_bps=args.dislocation_bps,
    )
    raw = events.off_hours_dislocation(frames, cfg)
    indexed = universe.set_index("instId")
    tickers = indexed.ticker.astype(str)
    raw = raw.assign(
        ticker=raw.inst_id.map(tickers),
        category=raw.inst_id.map(indexed.category),
    )
    if args.pool == "tech":
        raw = raw.loc[raw.ticker.isin(TECH)].copy()
    selected_universe = universe if args.pool == "all" else universe[universe.ticker.isin(TECH)]
    universe_counts = selected_universe.groupby("category").ticker.nunique().to_dict()
    if args.volume_ratio > 0:
        raw = raw.loc[raw.volume_ratio.fillna(0.0) >= args.volume_ratio].copy()
    if args.category != "all" and args.category not in set(raw.category.dropna()):
        raise ValueError(f"category not present in selected pool: {args.category}")
    raw = raw.sort_values("event_ts").reset_index(drop=True)

    rules = backtest.Rules(
        horizon="to_open",
        resolve_offset_minutes=args.resolve_offset_minutes,
        stop_loss_bps=args.stop_loss_bps,
        max_concurrent=args.slots,
        max_per_day=args.max_per_day,
        rank_column="abs_deviation",
        direction=args.direction,
        allow_short=True,
        min_trailing_volume=0.0,
    )
    rows = []
    selected = raw if args.category == "all" else raw[raw.category == args.category]
    category_groups = {args.category: selected}
    if args.category == "all":
        category_groups = {"all": selected}
        category_groups.update({str(category): group for category, group in raw.groupby("category")})
    for category, group in category_groups.items():
        # A category with fewer than four underlyings is descriptive only.
        if group.ticker.nunique() < 4:
            continue
        for period, part in [("train", group[group.event_ts < test_start]), ("test", group[group.event_ts >= test_start])]:
            result = backtest.run(part, frames, cfg, rules)
            _, metrics = portfolio.simulate(result.trades, 100_000.0, rules.max_concurrent)
            stats = backtest.summarize_trades(result.trades, rules)
            rows.append({
                "pool": args.pool,
                "category": category,
                "underlyings": group.ticker.nunique(),
                "universe_underlyings": int(len(selected_universe)) if category == "all" else int(universe_counts.get(category, 0)),
                "period": period,
                "events": len(part),
                "trades": len(result.trades),
                "return_pct": metrics.get("total_return", 0.0) * 100.0,
                "max_drawdown_pct": metrics.get("max_drawdown_pct", 0.0) * 100.0,
                "max_drawdown_recovery_days": metrics.get("max_drawdown_recovery_days", float("nan")),
                "longest_drawdown_recovery_days": metrics.get("longest_drawdown_recovery_days", float("nan")),
                "unrecovered_drawdown": metrics.get("unrecovered_drawdown", False),
                "sharpe_daily": metrics.get("sharpe_daily", float("nan")),
                "avg_net_bps": stats.get("avg_net_bps", float("nan")),
                "sqn": stats.get("sqn", float("nan")),
                "mean_profit_pvalue": stats.get("mean_profit_pvalue", float("nan")),
                "long_profit_pct": stats.get("long_profit_pct", float("nan")),
                "short_profit_pct": stats.get("short_profit_pct", float("nan")),
                "win_rate": stats.get("win_rate", float("nan")),
                "avg_hold_hours": stats.get("avg_hold_hours", float("nan")),
                "median_hold_hours": float(result.trades.hold_hours.median()) if not result.trades.empty else float("nan"),
                "long_trades": int((result.trades.side > 0).sum()) if not result.trades.empty else 0,
                "short_trades": int((result.trades.side < 0).sum()) if not result.trades.empty else 0,
            })

    raw.to_csv(result_dir / "events.csv", index=False)
    # Export the full-period trade list as well as train/test summaries so this
    # fixed rule can be joined to another strategy without reconstructing fills.
    full_result = backtest.run(raw, frames, cfg, rules)
    full_result.trades.to_csv(result_dir / "trades.csv", index=False)
    universe.to_csv(result_dir / "universe_categories.csv", index=False)
    pd.DataFrame(rows).to_csv(result_dir / "summary.csv", index=False)
    metadata = {
        "trade_metrics_note": METRIC_NOTE,
        "data_dir": str(data_dir),
        "pool": args.pool,
        "category": args.category,
        "test_start_utc": str(test_start),
        "dislocation_bps": args.dislocation_bps,
        "resolve_offset_minutes": args.resolve_offset_minutes,
        "volume_ratio": args.volume_ratio,
        "fee_bps": args.fee_bps,
        "slippage_bps": args.slippage_bps,
        "direction": args.direction,
        "slots": args.slots,
        "max_per_day": args.max_per_day,
        "stop_loss_bps": args.stop_loss_bps,
        "frames": len(frames),
        "events": len(raw),
        "note": "Research only; no order or live-trading path is imported.",
    }
    (result_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"输出目录: {result_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
