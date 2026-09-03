"""科技股双策略组合：普通 8-K 与财报 2.02 分流回测。"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from strategies.stocks.config import Config
from strategies.stocks.market import data
from strategies.stocks.market.universe_tech import TECH
from strategies.stocks.research import backtest, news_strategy, portfolio


def _write_reports(
    result_dir: Path, events: pd.DataFrame, trades: pd.DataFrame,
    metrics: dict, trade_stats: dict, rules: backtest.Rules,
    data_dir: str | Path,
) -> Path:
    """Write an immutable run bundle plus a latest convenience copy."""
    generated = datetime.now(timezone.utc)
    digest = hashlib.sha256()
    for frame in (events, trades):
        digest.update(frame.to_csv(index=False).encode())
    digest.update(json.dumps(metrics, sort_keys=True, default=str).encode())
    run_id = f"{generated.strftime('%Y%m%dT%H%M%S%fZ')}-{digest.hexdigest()[:12]}"
    run_dir = result_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    events.to_csv(run_dir / "hybrid_strategy_events.csv", index=False)
    trades.to_csv(run_dir / "hybrid_strategy_trades.csv", index=False)

    # Attach the route label to each fill for strategy-level diagnostics.
    labeled = trades.merge(
        events[["inst_id", "event_ts", "strategy"]],
        left_on=["inst_id", "entry_ts"], right_on=["inst_id", "event_ts"],
        how="left",
    )
    context = {
        "run_id": run_id,
        "generated_at_utc": generated.isoformat(),
    }
    combined_row = {"scope": "combined", **context, **metrics, "trades": len(trades)}
    # Keep portfolio metrics (especially Sharpe) authoritative.  Per-trade
    # diagnostics use a prefix when they share a name with portfolio fields.
    for key, value in trade_stats.items():
        combined_row[f"trade_{key}" if key in combined_row else key] = value
    rows = [combined_row]
    for strategy, group in labeled.groupby("strategy", dropna=False):
        if group.empty:
            continue
        rows.append({
            "scope": str(strategy),
            **context,
            "trades": int(len(group)),
            "win_rate": float((group.net > 0).mean()),
            "avg_net_bps": float(group.net.mean() * 1e4),
            "total_pnl": float(group.pnl.sum()),
            "stops": int((group.reason == "stop").sum()),
        })
    pd.DataFrame(rows).to_csv(run_dir / "report_long.csv", index=False)

    stability = []
    if not labeled.empty:
        labeled["period"] = labeled.entry_ts.map(
            lambda ts: "early" if ts < pd.Timestamp("2026-08-10", tz="UTC") else "late"
        )
        for period, group in labeled.groupby("period"):
            stability.append({
                "scope": "combined", **context,
                "breakdown": "period", "segment": period,
                "trades": int(len(group)), "win_rate": float((group.net > 0).mean()),
                "avg_net_bps": float(group.net.mean() * 1e4),
            })
        for strategy, group in labeled.groupby("strategy", dropna=False):
            for period, part in group.groupby("period"):
                stability.append({
                    "scope": str(strategy), **context,
                    "breakdown": "period", "segment": period,
                    "trades": int(len(part)), "win_rate": float((part.net > 0).mean()),
                    "avg_net_bps": float(part.net.mean() * 1e4),
                })
    pd.DataFrame(stability).to_csv(run_dir / "report_stability.csv", index=False)

    meta = {
        "run_id": run_id,
        "generated_at_utc": generated.isoformat(),
        "strategy": "hybrid_tech_8k_price_momentum",
        "description": "Non-LLM technology stock strategy; 2.02 filings route to earnings sleeve.",
        "sample_start_utc": str(events.accepted.min()) if not events.empty else None,
        "sample_end_utc": str(events.accepted.max()) if not events.empty else None,
        "event_count": int(len(events)),
        "trade_count": int(len(trades)),
        "universe": f"{os.environ.get('STOCK_POOL', 'tech').lower()} stocks, executable shorts only",
        "bar": "5m",
        "normal_route": {"observe_minutes": 90, "min_move_bps": 50, "item_202": False},
        "earnings_route": {"observe_minutes": 45, "min_move_bps": 0, "item_202": True},
        "execution": {"resolve_offset_minutes": rules.resolve_offset_minutes,
                       "stop_loss_bps": rules.stop_loss_bps,
                       "max_concurrent": rules.max_concurrent,
                       "max_per_day": rules.max_per_day},
        "metrics": metrics,
        "return_basis": "metrics.total_return is cumulative over the sample interval; annualized_return is a separate extrapolation",
        "trade_stats": trade_stats,
        "input_files": [str(Path(data_dir) / "sec_filings_raw.csv"),
                        str(Path(data_dir) / "universe.csv"),
                        str(Path(data_dir) / "5m" / "*.csv")],
        "notes": [
            "Direction comes from post-announcement token price, never from future prices.",
            "Results are an in-sample research run with a small event count; do not treat as forecast.",
        ],
    }
    (run_dir / "report_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    sample_start = events.accepted.min() if not events.empty else None
    sample_end = events.accepted.max() if not events.empty else None
    summary = (
        "# Hybrid Strategy Backtest\n\n"
        f"- Run: `{run_id}`\n"
        f"- Sample (accepted UTC): `{sample_start}` to `{sample_end}`\n"
        f"- Events: {len(events)} (normal 8-K: {len(events.loc[events.strategy == 'normal'])}; "
        f"2.02 earnings: {len(events.loc[events.strategy == 'earnings_2.02'])})\n"
        f"- Trades: {len(trades)}\n"
        f"- Interval return (not annualized): **{metrics.get('total_return', 0) * 100:+.2f}%**\n"
        f"- Max drawdown: **{metrics.get('max_drawdown_pct', 0) * 100:.2f}%**\n"
        f"- Daily Sharpe: **{metrics.get('sharpe_daily', float('nan')):.2f}**\n"
        f"- Win rate: **{trade_stats.get('win_rate', 0):.1%}**\n"
        f"- Average net: **{trade_stats.get('avg_net_bps', 0):+.1f}bp**\n\n"
        "## Execution\n\n"
        "- Normal 8-K: 90-minute observation, 50bp confirmation.\n"
        "- Item 2.02 earnings: 45-minute observation, no additional confirmation.\n"
        "- Shared limits: 600bp stop, exit 60 minutes after the next open, "
        "maximum 4 concurrent positions and 4 entries per day.\n\n"
        "## Files\n\n"
        "- `report_meta.json`: parameters, provenance and portfolio metrics.\n"
        "- `report_long.csv`: combined and route-level metrics.\n"
        "- `report_stability.csv`: early/late period stability.\n"
        "- `hybrid_strategy_events.csv` / `hybrid_strategy_trades.csv`: raw detail.\n\n"
        "> Research note: this is an in-sample run with 20 trades over a short window; "
        "it is not a live-performance forecast.\n"
    )
    (run_dir / "report.md").write_text(summary, encoding="utf-8")

    latest = result_dir / "latest"
    latest.mkdir(exist_ok=True)
    for path in run_dir.iterdir():
        (latest / path.name).write_bytes(path.read_bytes())
    return run_dir


def main() -> int:
    cfg = Config()
    result_dir = Path(cfg.result_dir)
    data_dir = Path(cfg.data_dir)
    input_dir = data_dir
    if not (input_dir / "sec_filings_raw.csv").exists() or not (input_dir / "universe.csv").exists():
        # Older external snapshots kept these two input files beside results.
        # The migrated project stores them under data/stocks, but accepting the
        # old layout keeps STOCK_ALPHA_ROOT useful as a data override.
        input_dir = result_dir
    universe = pd.read_csv(input_dir / "universe.csv")
    filings = pd.read_csv(input_dir / "sec_filings_raw.csv")
    filings["accepted"] = pd.to_datetime(
        filings.accepted, utc=True, errors="coerce", format="mixed"
    )
    filings = filings.dropna(subset=["accepted"]).drop_duplicates("accession")
    filings = filings.loc[filings.form.astype(str).str.startswith("8-K")]
    fallback_start = pd.Timestamp(
        os.environ.get("STOCK_STRATEGY_START", "2026-07-16"), tz="UTC"
    )
    if {"ticker", "list_ts"}.issubset(universe.columns):
        listed = pd.to_datetime(
            universe.set_index("ticker")["list_ts"], utc=True, errors="coerce"
        )
        min_start = filings["ticker"].map(listed) + pd.Timedelta(
            hours=cfg.listing_burn_in_hours
        )
        filings = filings.loc[filings.accepted >= min_start.fillna(fallback_start)]
    else:
        filings = filings.loc[filings.accepted >= fallback_start]
    pool_filter = os.environ.get("STOCK_POOL", "tech").lower()
    if pool_filter not in {"tech", "all"}:
        raise ValueError("STOCK_POOL must be tech or all")
    eligible_tickers = set(universe.ticker.astype(str)) if pool_filter == "all" else TECH
    side_filter = os.environ.get("STOCK_SIDE", "both").lower()
    if side_filter not in {"both", "long", "short"}:
        raise ValueError("STOCK_SIDE must be both, long, or short")
    inst_ids = sorted(filings.instId.dropna().unique())
    frames = data.to_bar_end(data.load_panel(inst_ids, "5m", cfg.data_dir), "5m")
    # 两条线使用不同的观察窗口，但最终在同一个组合回测里共享仓位限制。
    normal = news_strategy.build_events(
        filings, frames, observe_minutes=90, min_move_bps=50,
        max_stale_minutes=90, require_closed=True,
    )
    earnings = news_strategy.build_events(
        filings, frames, observe_minutes=45, min_move_bps=0,
        max_stale_minutes=90, require_closed=True,
    )
    normal = news_strategy.annotate_shortable(normal, universe)
    earnings = news_strategy.annotate_shortable(earnings, universe)

    def select(events: pd.DataFrame, is_earnings: bool) -> pd.DataFrame:
        events = events.loc[
            events.ticker.isin(eligible_tickers) & events.executable
        ].copy()
        has_202 = events["items"].astype(str).str.contains("2.02")
        return events.loc[has_202 if is_earnings else ~has_202]

    normal = select(normal, False)
    earnings = select(earnings, True)
    if side_filter == "long":
        normal = normal.loc[normal.side > 0].copy()
        earnings = earnings.loc[earnings.side > 0].copy()
    elif side_filter == "short":
        normal = normal.loc[normal.side < 0].copy()
        earnings = earnings.loc[earnings.side < 0].copy()
    normal["strategy"] = "normal"
    earnings["strategy"] = "earnings_2.02"
    events = pd.concat([normal, earnings], ignore_index=True).sort_values("event_ts")
    rules = backtest.Rules(
        horizon="to_open", stop_loss_bps=600, resolve_offset_minutes=60,
        min_trailing_volume=0, rank_column="abs_move_bps",
        max_concurrent=4, max_per_day=4, allow_short=True,
    )
    result = backtest.run(events, frames, cfg, rules)
    _, metrics = portfolio.simulate(result.trades, 100_000, 4)
    trade_stats = backtest.summarize_trades(result.trades, rules)
    run_dir = _write_reports(result_dir, events, result.trades, metrics, trade_stats, rules, input_dir)
    trades = result.trades
    print(f"普通 8-K: {len(normal)} 条候选；2.02 财报: {len(earnings)} 条候选")
    print(f"组合成交 {len(trades)} 笔，胜率 {(trades.net > 0).mean():.1%}，"
          f"平均净收益 {trades.net.mean() * 1e4:+.0f}bp，总收益 {metrics['total_return'] * 100:+.2f}%，"
          f"最大回撤 {metrics['max_drawdown_pct'] * 100:.2f}%，"
          f"Sharpe {metrics['sharpe_daily']:.2f}，盈亏比 "
          f"{backtest.summarize_trades(trades, rules)['profit_factor']:.2f}")
    print(f"报告目录: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
