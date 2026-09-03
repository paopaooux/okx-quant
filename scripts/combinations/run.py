"""Combine the stock hybrid sleeve with the OKX-quant crypto sleeve.

The two strategies have different execution clocks and position rules.  This
script therefore combines normalized daily equity curves, rather than simply
concatenating their trade lists.  The comparison window is the stock sleeve's
available sample; crypto trades crossing the window boundary are excluded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from strategies.stocks.research import portfolio

BAR_MINUTES = 15
QUANT_SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")


def _quant_trades(oos: pd.DataFrame, tail: float, cost_bps: float) -> pd.DataFrame:
    """Reproduce okx-quant's c/tail selection from its OOS file."""
    tail_name = str(tail).rstrip("0").rstrip(".")
    hi_col, lo_col = f"hi_{tail_name}", f"lo_{tail_name}"
    required = {"symbol", "ts", "dt", "p_up", hi_col, lo_col, "held_c", "exit_ret_c"}
    missing = required.difference(oos.columns)
    if missing:
        raise ValueError(f"okx-quant OOS file is missing columns: {sorted(missing)}")
    rows: list[dict] = []
    for symbol, group in oos.groupby("symbol", sort=True):
        group = group.sort_values("ts").reset_index(drop=True)
        free_at = 0
        for i, row in group.iterrows():
            held = row["held_c"]
            exit_ret = row["exit_ret_c"]
            if i < free_at or not np.isfinite(held) or not np.isfinite(exit_ret):
                continue
            if row["p_up"] >= row[hi_col]:
                side = 1
            elif row["p_up"] <= row[lo_col]:
                side = -1
            else:
                continue
            gross = side * float(exit_ret)
            rows.append({
                "strategy": "okx_quant_c_tail_0.01",
                "symbol": symbol,
                "entry_ts": row["dt"],
                "exit_ts": row["dt"] + pd.Timedelta(minutes=(float(held) + 1) * BAR_MINUTES),
                "side": side,
                "gross": gross,
                "net": gross - cost_bps / 1e4,
                "hold_hours": (float(held) + 1) * BAR_MINUTES / 60,
                "signal_strength": abs(float(row["p_up"]) - 0.5),
            })
            free_at = i + int(held) + 1
    return pd.DataFrame(rows)

def _align_curve(curve: pd.Series, days: pd.DatetimeIndex) -> pd.Series:
    if curve.empty:
        return pd.Series(1.0, index=days)
    curve = curve.sort_index()
    curve = curve / float(curve.iloc[0])
    daily = curve.resample("1D").last().ffill()
    return daily.reindex(days).ffill().fillna(1.0)


def _quant_curve(trades: pd.DataFrame, days: pd.DatetimeIndex) -> pd.Series:
    """Three equal crypto sleeves, matching okx-quant's portfolio.py."""
    total = pd.Series(0.0, index=days)
    for symbol in QUANT_SYMBOLS:
        group = trades.loc[trades.symbol == symbol].sort_values("exit_ts")
        running = 1.0
        values = []
        for day in days:
            exits = group.loc[group.exit_ts.dt.normalize() == day, "net"]
            for net in exits:
                running *= 1.0 + float(net)
            values.append(running)
        total += pd.Series(values, index=days) / len(QUANT_SYMBOLS)
    return total


def _pooled_curve(
    trades: pd.DataFrame, days: pd.DatetimeIndex, policy: int | str,
) -> tuple[pd.Series, pd.DataFrame, int]:
    """Run fixed or dynamic shared-pool sizing without future information.

    Fixed policies use one equal slice per slot.  Dynamic policies can accept
    either two or three positions, but reserve the three-slot budget from the
    outset (one third per accepted trade).  That keeps the dynamic variants
    from becoming leveraged when a third entry arrives; resizing open trades
    would require an intrabar mark that is not present in the trade exports.
    """
    if isinstance(policy, int):
        if policy < 1:
            raise ValueError("max_slots must be at least 1")
    elif policy not in ("dynamic_2_to_3", "dynamic_3_to_2"):
        raise ValueError(f"unknown pool policy: {policy}")
    ordered = trades.sort_values(["entry_ts", "strategy"]).reset_index(drop=True)
    events: list[tuple[pd.Timestamp, int, str, int]] = []
    for i, row in ordered.iterrows():
        # Exits are processed before entries at the same timestamp.
        events.append((row.entry_ts, 1, "in", i))
        events.append((row.exit_ts, 0, "out", i))
    events.sort(key=lambda item: (item[0], item[1]))
    equity = 1.0
    notional: dict[int, float] = {}
    accepted: list[int] = []
    skipped = 0
    position_budget = 1.0 / policy if isinstance(policy, int) else 1.0 / 3.0
    values = []
    for day in days:
        for stamp, _, kind, idx in events:
            if stamp.normalize() != day:
                continue
            row = ordered.iloc[idx]
            if kind == "out":
                if idx in notional:
                    equity += notional.pop(idx) * float(row.net)
            else:
                active_limit = policy if isinstance(policy, int) else 3
                if policy == "dynamic_2_to_3":
                    active_strategies = {ordered.iloc[i].strategy for i in notional}
                    # Base capacity is two.  A third slot is only used when it
                    # diversifies the active strategy mix.
                    active_limit = 3 if len(notional) >= 2 and row.strategy not in active_strategies else 2
                elif policy == "dynamic_3_to_2":
                    # Capacity three is reserved for a diversified book.  If
                    # the two existing positions are from one sleeve, keep a
                    # two-slot cap unless the new entry adds the missing
                    # sleeve.  This uses only information known at entry.
                    counts = {name: sum(ordered.iloc[i].strategy == name for i in notional)
                              for name in {ordered.iloc[i].strategy for i in notional}}
                    if len(notional) < 2:
                        active_limit = 3
                    elif len(counts) == 1:
                        active_limit = 3 if row.strategy not in counts else 2
                    else:
                        active_limit = 3 if counts.get(row.strategy, 0) == min(counts.values()) else 2
                if len(notional) < active_limit:
                    notional[idx] = equity * position_budget
                    ordered.loc[idx, "position_weight"] = position_budget
                    accepted.append(idx)
                else:
                    skipped += 1
        values.append(equity)
    return pd.Series(values, index=days), ordered.iloc[accepted].copy(), skipped


def _stats(curve: pd.Series, trades: pd.DataFrame, days: pd.DatetimeIndex) -> dict:
    returns = curve.pct_change().dropna()
    total = float(curve.iloc[-1] - 1.0)
    span_days = max((days[-1] - days[0]).total_seconds() / 86400.0, 1.0)
    return {
        "return_basis": "sample_interval_cumulative",
        "sample_days": span_days,
        "total_return": total,
        "annualized_return": float((1 + total) ** (365.25 / span_days) - 1) if total > -1 else -1.0,
        "max_drawdown_pct": float((curve / curve.cummax() - 1).min()),
        "sharpe_daily": float(returns.mean() / returns.std(ddof=1) * np.sqrt(365))
        if returns.std(ddof=1) > 0 else np.nan,
        "vol_annualized": float(returns.std(ddof=1) * np.sqrt(365)) if len(returns) > 1 else np.nan,
        "n_trades": int(len(trades)),
        "win_rate": float((trades.net > 0).mean()) if not trades.empty else np.nan,
        "avg_net_bps": float(trades.net.mean() * 1e4) if not trades.empty else np.nan,
        # Portfolio-level average exposure: each trade contributes its weight
        # for the hours it is open.  This is the fraction of total capital
        # deployed on average, rather than the size of one slot.
        "avg_position_pct": float(
            (trades.position_weight * trades.hold_hours).sum() / (span_days * 24.0)
        ) if "position_weight" in trades and "hold_hours" in trades and not trades.empty else np.nan,
        "avg_hold_hours": float(trades.hold_hours.mean()) if "hold_hours" in trades and not trades.empty else np.nan,
        "avg_concurrent": float(trades.hold_hours.sum() / (span_days * 24.0)) if "hold_hours" in trades and not trades.empty else np.nan,
    }


def _write_bundle(
    output_root: Path,
    stock: pd.DataFrame,
    quant: pd.DataFrame,
    daily: pd.DataFrame,
    reports: pd.DataFrame,
    meta: dict,
) -> Path:
    digest = hashlib.sha256()
    for frame in (stock, quant, daily, reports):
        digest.update(frame.to_csv(index=False).encode())
    generated = datetime.now(timezone.utc)
    run_id = f"{generated.strftime('%Y%m%dT%H%M%S%fZ')}-{digest.hexdigest()[:12]}"
    run_dir = output_root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    stock.to_csv(run_dir / "stock_trades.csv", index=False)
    quant.to_csv(run_dir / "quant_trades.csv", index=False)
    pd.concat([
        stock.assign(strategy="xstock_hybrid", symbol=stock.inst_id),
        quant,
    ], ignore_index=True, sort=False).to_csv(run_dir / "combined_trades.csv", index=False)
    daily.to_csv(run_dir / "component_daily.csv", index=False)
    reports.assign(run_id=run_id).to_csv(run_dir / "report_long.csv", index=False)
    meta = {**meta, "run_id": run_id, "generated_at_utc": generated.isoformat()}
    (run_dir / "report_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    combo = reports.loc[reports.scope == "combo_50_stock"].iloc[0]
    pooled = reports.loc[reports.scope == "pooled_3_slots"].iloc[0]
    allocation_rows = []
    for _, row in reports.loc[reports["mode"] == "static"].iterrows():
        allocation_rows.append(
            f"| {row.stock_weight:.0%} | {row.total_return:+.2%} | "
            f"{row.max_drawdown_pct:.2%} | {row.sharpe_daily:.2f} | "
            f"{row.stock_win_rate:.1%} | {row.quant_win_rate:.1%} |"
        )
    summary = (
        "# Cross-Strategy Combination Backtest\n\n"
        f"- Run: `{run_id}`\n"
        f"- Window (UTC): `{meta['sample_start_utc']}` to `{meta['sample_end_utc']}`\n"
        f"- Stock trades: {len(stock)}; crypto trades: {len(quant)}\n"
        f"- Quant OOS data through: `{meta['quant_oos_end_utc']}`; last quant signal: "
        f"`{meta['quant_signal_end_utc']}`\n"
        "- Primary result: fixed shared-pool comparison with 1/2/3/4 slots\n"
        f"- Static 50/50 reference return (not annualized): **{combo.total_return * 100:+.2f}%**\n"
        f"- Static 50/50 sleeve win rates: stocks **{combo.stock_win_rate:.1%}**, "
        f"crypto **{combo.quant_win_rate:.1%}**\n\n"
        "## Reference allocations\n\n"
        "| Stock weight | Interval return | Max drawdown | Daily Sharpe | Stock win rate | Quant win rate |\n"
        "|---:|---:|---:|---:|---:|---:|\n"
        + "\n".join(allocation_rows)
        + "\n\n"
        "## Shared-pool strategies\n\n"
        "| Strategy | Interval return | Max drawdown | Daily Sharpe | Trades | Trade win rate | Avg net (bp) | Avg total exposure | Avg hold (h) | Avg concurrent | Skipped entries |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
        + "\n".join(
            f"| {row.scope} | {row.total_return:+.2%} | "
            f"{row.max_drawdown_pct:.2%} | {row.sharpe_daily:.2f} | "
            f"{row.n_trades:.0f} | {row.win_rate:.1%} | {row.avg_net_bps:+.1f} | "
            f"{row.avg_position_pct:.1%} | {row.avg_hold_hours:.1f} | "
            f"{row.avg_concurrent:.2f} | "
            f"{row.skipped_entries:.0f} |"
            for _, row in reports.loc[reports["mode"] == "pooled"].iterrows()
        )
        + "\n\n"
        "## Interpretation\n\n"
        "The crypto sleeve uses okx-quant config c, tail 0.01, both directions, "
        "10bp cost. The stock sleeve uses the latest hybrid result. Crypto trades "
        "that cross the window boundary are excluded, so this is a fair overlap test.\n\n"
        f"The fixed 3-slot pool returns **{pooled.total_return:+.2%}** with "
        f"**{pooled.max_drawdown_pct:.2%}** drawdown. The shared-pool table is the "
        "primary combination result; static rows are reference allocations only.\n\n"
        "> Research note: the overlap contains only a small number of crypto trades; "
        "the combined result is exploratory, not a live-performance forecast.\n"
    )
    (run_dir / "report.md").write_text(summary, encoding="utf-8")

    latest = output_root / "latest"
    latest.mkdir(parents=True, exist_ok=True)
    for path in run_dir.iterdir():
        (latest / path.name).write_bytes(path.read_bytes())
    return run_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--okx-quant-root", type=Path, default=PROJECT_ROOT,
                        help="虚拟币项目根目录；默认使用当前运行目录")
    parser.add_argument("--stock-trades", type=Path,
                        default=PROJECT_ROOT / "results/stocks/latest/hybrid_strategy_trades.csv")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results/combinations")
    parser.add_argument("--tail", type=float, default=0.01)
    parser.add_argument("--cost-bps", type=float, default=10.0)
    parser.add_argument("--stock-weight", type=float, default=None,
                        help="Add a custom static stock weight to the comparison grid")
    parser.add_argument("--pool-slots", type=int, default=None,
                        help="Add a custom fixed shared-pool slot count to the comparison grid")
    args = parser.parse_args()

    stock = pd.read_csv(args.stock_trades)
    for col in ("entry_ts", "exit_ts"):
        stock[col] = pd.to_datetime(stock[col], utc=True)
    stock_start, stock_end = stock.entry_ts.min(), stock.exit_ts.max()
    oos_path = args.okx_quant_root / "results" / "crypto" / "oos_dir_c.csv.gz"
    oos = pd.read_csv(oos_path)
    oos["dt"] = pd.to_datetime(oos["dt"], utc=True)
    quant_all = _quant_trades(oos, args.tail, args.cost_bps)
    quant = quant_all.loc[
        (quant_all.entry_ts >= stock_start) & (quant_all.exit_ts <= stock_end)
    ].copy()
    days = pd.date_range(stock_start.normalize(), stock_end.normalize(), freq="D", tz="UTC")
    stock_curve, _ = portfolio.simulate(stock, 1.0, 4)
    stock_daily = _align_curve(stock_curve, days)
    quant_daily = _quant_curve(quant, days)
    daily = pd.DataFrame({
        "date": days,
        "stock_equity": stock_daily.to_numpy(),
        "quant_equity": quant_daily.to_numpy(),
    })
    reports = []
    stock_win_rate = float((stock.net > 0).mean()) if not stock.empty else np.nan
    stock_avg_net = float(stock.net.mean() * 1e4) if not stock.empty else np.nan
    quant_win_rate = float((quant.net > 0).mean()) if not quant.empty else np.nan
    quant_avg_net = float(quant.net.mean() * 1e4) if not quant.empty else np.nan
    # Keep only the reference rows used by the operational comparison:
    # crypto-only, static 50/50, and stocks-only.  The main result is the
    # shared-pool 1/2/3/4-slot table below.
    stock_weights = [0.0, 0.5, 1.0]
    if args.stock_weight is not None:
        if not 0.0 <= args.stock_weight <= 1.0:
            raise SystemExit("--stock-weight must be between 0 and 1")
        stock_weights.append(args.stock_weight)
    for stock_weight in sorted(set(stock_weights)):
        combo_curve = stock_weight * stock_daily + (1 - stock_weight) * quant_daily
        # Win rate and average bps are sleeve-level quantities.  For a mixed
        # row they would depend on arbitrary trade counting, so leave them
        # empty and expose both sleeve values explicitly below.
        combo_trades = stock if stock_weight == 1 else quant if stock_weight == 0 else pd.DataFrame()
        row = _stats(combo_curve, combo_trades, days)
        row.update(scope=f"combo_{int(stock_weight * 100)}_stock", mode="static",
                   stock_weight=stock_weight, quant_weight=1 - stock_weight,
                   stock_trades=len(stock), quant_trades=len(quant),
                   stock_win_rate=stock_win_rate, stock_avg_net_bps=stock_avg_net,
                   quant_win_rate=quant_win_rate, quant_avg_net_bps=quant_avg_net)
        reports.append(row)
        if stock_weight in (0.0, 0.5, 1.0):
            daily[f"equity_{int(stock_weight * 100)}_stock"] = combo_curve.to_numpy()
    report_frame = pd.DataFrame(reports)
    daily["combo_50_stock"] = (0.5 * stock_daily + 0.5 * quant_daily).to_numpy()
    all_trades = pd.concat([
        stock.assign(strategy="xstock_hybrid", symbol=stock.inst_id), quant,
    ], ignore_index=True, sort=False)
    pool_specs = [(slots, f"pooled_{slots}_slots") for slots in (1, 2, 3, 4)]
    if args.pool_slots is not None:
        if args.pool_slots < 1:
            raise SystemExit("--pool-slots must be at least 1")
        pool_specs.append((args.pool_slots, f"pooled_{args.pool_slots}_slots"))
    for policy, scope in pool_specs:
        pool_curve, executed, skipped = _pooled_curve(all_trades, days, policy)
        pool_row = _stats(pool_curve, executed, days)
        pool_row.update(scope=scope, mode="pooled",
                        stock_weight=np.nan, quant_weight=np.nan,
                        pool_slots=policy if isinstance(policy, int) else np.nan,
                        skipped_entries=skipped,
                        stock_trades=len(stock), quant_trades=len(quant),
                        stock_win_rate=stock_win_rate, stock_avg_net_bps=stock_avg_net,
                        quant_win_rate=quant_win_rate, quant_avg_net_bps=quant_avg_net)
        reports.append(pool_row)
        daily[scope] = pool_curve.to_numpy()
    report_frame = pd.DataFrame(reports)
    meta = {
        "strategy": "xstock_hybrid_plus_okx_quant",
        "description": "Daily equity-curve combination over the shared stock sample window.",
        "sample_start_utc": stock_start.isoformat(),
        "sample_end_utc": stock_end.isoformat(),
        "stock_input": str(args.stock_trades),
        "quant_input": str(oos_path),
        "quant_oos_end_utc": oos["dt"].max().isoformat(),
        "quant_signal_end_utc": quant_all["entry_ts"].max().isoformat() if not quant_all.empty else None,
        "quant_selection": {"config": "c", "tail": args.tail, "policy": "both", "cost_bps": args.cost_bps},
        "stock_allocation_grid": sorted(set(stock_weights)),
        "shared_pool_strategies": [name for _, name in pool_specs],
        "stock_position_rule": "hybrid strategy portfolio, max 4 concurrent positions",
        "quant_position_rule": "three equal symbol sleeves, one position per symbol",
        "annualization": "daily Sharpe uses sqrt(365) because the combined portfolio includes a 24/7 crypto sleeve",
        "return_basis": "total_return is cumulative over the sample interval; annualized_return is a separate extrapolation",
    }
    run_dir = _write_bundle(args.output, stock, quant, daily, report_frame, meta)
    combo = report_frame.loc[report_frame.scope == "combo_50_stock"].iloc[0]
    pool_rows = report_frame.loc[report_frame.scope.isin(
        [f"pooled_{slots}_slots" for slots in (1, 2, 3, 4)]
    )].sort_values("pool_slots")
    print(f"共同窗口: {stock_start} -> {stock_end}")
    print(f"股票 {len(stock)} 笔；okx-quant {len(quant)} 笔（c / tail={args.tail:g} / {args.cost_bps:g}bp）")
    print(f"静态 50/50：收益 {combo.total_return * 100:+.2f}%，最大回撤 {combo.max_drawdown_pct * 100:.2f}%，"
          f"Sharpe {combo.sharpe_daily:.2f}，胜率（股票/crypto）"
          f" {combo.stock_win_rate:.1%}/{combo.quant_win_rate:.1%}")
    for _, row in pool_rows.iterrows():
        print(f"共享资金池 {int(row.pool_slots)} 槽：{int(row.n_trades)} 笔，"
              f"收益 {row.total_return * 100:+.2f}%，最大回撤 {row.max_drawdown_pct * 100:.2f}%，"
              f"Sharpe {row.sharpe_daily:.2f}")
    print(f"报告目录: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
