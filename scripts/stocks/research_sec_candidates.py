"""Reproducible SEC/OKX stock-perpetual candidate study.

The script deliberately keeps the rule fixed and reports a time split instead
of selecting the best result from the full sample.  It is research-only: no
exchange client or order path is imported.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from strategies.stocks.config import Config
from strategies.stocks.market import data
from strategies.stocks.research import backtest, news_strategy, portfolio


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=os.environ.get("STOCK_ALPHA_DATA_DIR", "data/stocks_swap"))
    parser.add_argument("--result-dir", default=os.environ.get("STOCK_ALPHA_RESULT_DIR", "results/stocks_sec_research"))
    parser.add_argument("--test-start", default="2026-08-01", help="UTC date starting the untouched test period")
    parser.add_argument("--observe-minutes", type=float, default=30.0)
    parser.add_argument("--min-move-bps", type=float, default=100.0)
    parser.add_argument("--hold-hours", type=float, default=1.0)
    parser.add_argument("--fee-bps", type=float, default=10.0)
    parser.add_argument("--slippage-bps", type=float, default=12.0)
    return parser.parse_args()


def _load(data_dir: Path) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    filings = pd.read_csv(data_dir / "sec_filings_raw.csv")
    filings["accepted"] = pd.to_datetime(filings.accepted, utc=True, errors="coerce", format="mixed")
    filings = filings.dropna(subset=["accepted"]).drop_duplicates("accession")
    filings = filings.loc[filings.form.astype(str).isin(["8-K", "8-K/A"])].copy()
    inst_ids = sorted(filings.instId.dropna().astype(str).unique())
    frames = data.to_bar_end(
        data.load_panel(inst_ids, "5m", data_dir), "5m"
    )
    return filings, frames


def _event_returns(
    events: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
    hold_hours: float,
    round_trip_cost: float,
) -> pd.DataFrame:
    rows: list[dict] = []
    for event in events.itertuples(index=False):
        frame = frames.get(event.inst_id)
        if frame is None:
            continue
        entry_pos = frame.index.searchsorted(event.event_ts, side="right") - 1
        exit_pos = frame.index.searchsorted(
            event.event_ts + pd.Timedelta(hours=hold_hours), side="right"
        ) - 1
        if entry_pos < 0 or exit_pos <= entry_pos:
            continue
        entry = float(frame["close"].iloc[entry_pos])
        exit_price = float(frame["close"].iloc[exit_pos])
        if not (np.isfinite(entry) and np.isfinite(exit_price)) or entry <= 0:
            continue
        gross = int(event.side) * (exit_price / entry - 1.0)
        rows.append({
            "inst_id": event.inst_id,
            "event_ts": event.event_ts,
            "event_day": event.event_day,
            "route": event.route,
            "side": int(event.side),
            "gross": gross,
            "net": gross - round_trip_cost,
        })
    return pd.DataFrame(rows)


def _summary(returns: pd.DataFrame, test_start: pd.Timestamp) -> pd.DataFrame:
    if returns.empty:
        return pd.DataFrame()
    work = returns.copy()
    work["period"] = np.where(work.event_ts < test_start, "train", "test")
    rows = []
    for (route, period), group in work.groupby(["route", "period"], dropna=False):
        daily = group.groupby("event_day").net.mean()
        rows.append({
            "route": route,
            "period": period,
            "events": len(group),
            "event_days": group.event_day.nunique(),
            "net_bps": group.net.mean() * 1e4,
            "win_rate": (group.net > 0).mean(),
            "daily_net_bps": daily.mean() * 1e4,
        })
    return pd.DataFrame(rows).sort_values(["route", "period"]).reset_index(drop=True)


def main() -> int:
    args = _args()
    data_dir = Path(args.data_dir)
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    test_start = pd.Timestamp(args.test_start, tz="UTC")
    cfg = Config(data_dir=str(data_dir), fee_bps=args.fee_bps, slippage_bps=args.slippage_bps)
    filings, frames = _load(data_dir)
    events = news_strategy.build_events(
        filings,
        frames,
        observe_minutes=args.observe_minutes,
        min_move_bps=args.min_move_bps,
        max_stale_minutes=90.0,
        require_closed=True,
    )
    if events.empty:
        raise RuntimeError("没有生成 SEC 事件；请检查 SEC CSV 与 5m 行情覆盖")
    events = events.copy()
    events["route"] = np.where(
        events["items"].astype(str).str.contains("2.02"), "item_2.02", "other_8k"
    )
    returns = _event_returns(events, frames, args.hold_hours, cfg.round_trip_bps / 1e4)
    summary = _summary(returns, test_start)

    rules = backtest.Rules(
        horizon="fixed",
        max_hold_hours=args.hold_hours,
        stop_loss_bps=600.0,
        max_concurrent=4,
        max_per_day=4,
        rank_column="abs_move_bps",
        allow_short=True,
        min_trailing_volume=0.0,
    )
    portfolio_rows = []
    for route, group in [("all", events), ("item_2.02", events[events.route == "item_2.02"])]:
        for period, part in [("train", group[group.event_ts < test_start]), ("test", group[group.event_ts >= test_start])]:
            result = backtest.run(part, frames, cfg, rules)
            _, metrics = portfolio.simulate(result.trades, 100_000.0, rules.max_concurrent)
            portfolio_rows.append({
                "route": route,
                "period": period,
                "events": len(part),
                "trades": len(result.trades),
                "return_pct": metrics.get("total_return", 0.0) * 100.0,
                "max_drawdown_pct": metrics.get("max_drawdown_pct", 0.0) * 100.0,
                "max_drawdown_recovery_days": metrics.get("max_drawdown_recovery_days", float("nan")),
                "longest_drawdown_recovery_days": metrics.get("longest_drawdown_recovery_days", float("nan")),
                "unrecovered_drawdown": metrics.get("unrecovered_drawdown", False),
                "sharpe_daily": metrics.get("sharpe_daily", np.nan),
                "avg_hold_hours": result.stats.get("avg_hold_hours", np.nan),
                "median_hold_hours": float(result.trades.hold_hours.median()) if not result.trades.empty else np.nan,
                "long_trades": int((result.trades.side > 0).sum()) if not result.trades.empty else 0,
                "short_trades": int((result.trades.side < 0).sum()) if not result.trades.empty else 0,
            })

    events.to_csv(result_dir / "events.csv", index=False)
    returns.to_csv(result_dir / "event_returns.csv", index=False)
    summary.to_csv(result_dir / "event_summary.csv", index=False)
    pd.DataFrame(portfolio_rows).to_csv(result_dir / "portfolio_summary.csv", index=False)
    metadata = {
        "data_dir": str(data_dir),
        "test_start_utc": str(test_start),
        "observe_minutes": args.observe_minutes,
        "min_move_bps": args.min_move_bps,
        "hold_hours": args.hold_hours,
        "fee_bps": args.fee_bps,
        "slippage_bps": args.slippage_bps,
        "frames": len(frames),
        "events": len(events),
        "note": "Research only; no order or live-trading path is imported.",
    }
    (result_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(pd.DataFrame(portfolio_rows).to_string(index=False))
    print(f"输出目录: {result_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
