"""事件驱动回测。

与常见的调仓型回测不同，这里没有"每 15 分钟看一眼要不要换仓"。
只有事件触发时才开仓，仓位到收敛点或止损离场。这样回测的自由度
只剩下几个明确的风控参数，而不是一整片可以搜的网格。

"少出手"是硬约束而不是结果：每天最多几笔、同时最多几个仓位，
在信号排队的时候按事件强度取前几名，其余全部放弃。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import Config


@dataclass
class Rules:
    horizon: str = "to_open"          # 持有到收敛点
    max_hold_hours: float = 30.0
    stop_loss_bps: float = 250.0      # 单笔止损
    take_profit_bps: float = 0.0      # 0 = 不设止盈，让收敛自然完成
    position_notional: float = 1_000.0
    max_concurrent: int = 3
    max_per_day: int = 3
    rank_column: str = "abs_signal"   # 信号排队时按强度取前几名
    resolve_offset_minutes: float = 30.0  # 收敛点之后多久离场
    direction: str = "both"            # both / long / short
    allow_short: bool = True
    min_trailing_volume: float = 150_000.0


@dataclass
class Result:
    trades: pd.DataFrame
    stats: dict
    equity: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))


def _signal_strength(events: pd.DataFrame) -> pd.Series:
    """事件强度，用于同一时刻多个信号排队时取前几名。

    不同事件类型的强度列不同名，统一成一个绝对值列。

    "unabsorbed" 必须排在最前面。btc_leadlag 事件的 btc_move 在同一天
    对整个板块是同一个数，如果排序落到它上面，所有名额都会被并列打平
    后的第一个标的拿走——实测就是这样：每天最多一笔的版本 13 笔全部
    落在 XMSTR，而 XMSTR 恰好是逐标的检验里斜率 t≈0.004 的那一个。
    unabsorbed = beta*btc_move - token_move 是逐标的的量，也正是机制
    本身说的"还没被消化的部分"。
    """
    for column in ("unabsorbed", "residual_deviation", "deviation", "gap", "shock"):
        if column in events.columns and events[column].notna().any():
            return events[column].abs().fillna(0.0)
    return pd.Series(1.0, index=events.index)


def _exit_scan(
    frame: pd.DataFrame, entry_ts: pd.Timestamp, entry_price: float, side: int,
    deadline: pd.Timestamp, stop_loss: float, take_profit: float,
) -> tuple[pd.Timestamp, float, str]:
    """逐根 bar 向前扫，先命中止损/止盈就在那里离场。

    止损用 bar 内的最高/最低价判断而不是收盘价：只看收盘价会让
    回测系统性地漏掉真实发生过的止损，把亏损单变成盈利单。
    这一步在薄盘代币上尤其重要，它们的影线经常很长。
    """
    window = frame.loc[(frame.index > entry_ts) & (frame.index <= deadline)]
    if window.empty:
        return entry_ts, entry_price, "no_data"
    for stamp, row in window.iterrows():
        if side > 0:
            adverse = (float(row["low"]) / entry_price - 1.0)
            favorable = (float(row["high"]) / entry_price - 1.0)
        else:
            adverse = -(float(row["high"]) / entry_price - 1.0)
            favorable = -(float(row["low"]) / entry_price - 1.0)
        # 同一根 bar 里止损和止盈都被触及时，保守地认定止损先成交。
        if stop_loss > 0 and adverse <= -stop_loss:
            stop_price = (
                entry_price * (1 - stop_loss) if side > 0 else entry_price * (1 + stop_loss)
            )
            return stamp, stop_price, "stop"
        if take_profit > 0 and favorable >= take_profit:
            return stamp, (
                entry_price * (1 + take_profit) if side > 0 else entry_price * (1 - take_profit)
            ), "target"
    last = window.iloc[-1]
    return window.index[-1], float(last["close"]), "deadline"


def run(
    events: pd.DataFrame, frames: dict[str, pd.DataFrame], config: Config,
    rules: Rules | None = None, slippage_bps: float | None = None,
) -> Result:
    """按事件顺序模拟一遍。"""
    rules = rules or Rules()
    slippage_bps = config.slippage_bps if slippage_bps is None else slippage_bps
    if events.empty:
        return Result(pd.DataFrame(), {"n_trades": 0})

    queue = events.copy()
    queue["abs_signal"] = _signal_strength(queue)
    # 休市偏离的机制是按偏离幅度排队，而不是按正负号排队。保留一个
    # 明确的列，避免把有符号 deviation 误当作强度。
    if "deviation" in queue.columns:
        queue["abs_deviation"] = queue["deviation"].abs()
    if "trailing_quote_volume_24h" in queue.columns:
        queue = queue.loc[
            queue.trailing_quote_volume_24h.fillna(0) >= rules.min_trailing_volume
        ]
    if rules.direction not in {"both", "long", "short"}:
        raise ValueError("rules.direction must be one of: both, long, short")
    if rules.direction == "long" or (rules.direction == "both" and not rules.allow_short):
        queue = queue.loc[queue.side > 0]
    elif rules.direction == "short":
        queue = queue.loc[queue.side < 0]
    if rules.rank_column in queue.columns and len(queue) > 1:
        grouped = queue.groupby("event_ts")[rules.rank_column]
        multi = grouped.transform("size") > 1
        spread = grouped.transform("std").fillna(0.0)
        if multi.any() and float(spread.loc[multi].max()) == 0.0:
            # 排序列在每个时点内没有差异，取前几名等于随机取，而"随机"在
            # pandas 里是稳定的原始顺序，会一直挑同一个标的。
            print(f"[backtest] 警告: 排序列 {rules.rank_column} 在时点内无差异，"
                  f"名额分配退化为固定顺序")
    sort_columns = ["event_ts", rules.rank_column]
    ascending = [True, False]
    if "inst_id" in queue.columns:
        sort_columns.append("inst_id")
        ascending.append(True)
    queue = queue.sort_values(sort_columns, ascending=ascending)

    entry_cost = (config.fee_bps + slippage_bps) / 1e4
    exit_cost = (config.fee_bps + slippage_bps) / 1e4
    stop_loss = rules.stop_loss_bps / 1e4
    take_profit = rules.take_profit_bps / 1e4

    open_positions: list[tuple[pd.Timestamp, str]] = []
    per_day: dict[pd.Timestamp, int] = {}
    trades = []

    for event in queue.itertuples(index=False):
        now = event.event_ts
        open_positions = [p for p in open_positions if p[0] > now]
        if len(open_positions) >= rules.max_concurrent:
            continue
        day = pd.Timestamp(now.date())
        if per_day.get(day, 0) >= rules.max_per_day:
            continue
        if any(inst == event.inst_id for _, inst in open_positions):
            continue
        frame = frames.get(event.inst_id)
        if frame is None:
            continue
        position = frame.index.searchsorted(now, side="right") - 1
        if position < 0:
            continue
        entry_price = float(frame["close"].iloc[position])
        if not np.isfinite(entry_price) or entry_price <= 0:
            continue

        resolve_ts = getattr(event, "resolve_ts", None)
        if rules.horizon == "to_open" and resolve_ts is not None and pd.notna(resolve_ts):
            deadline = pd.Timestamp(resolve_ts) + pd.Timedelta(minutes=rules.resolve_offset_minutes)
        else:
            deadline = now + pd.Timedelta(hours=rules.max_hold_hours)
        deadline = min(deadline, now + pd.Timedelta(hours=rules.max_hold_hours))
        if deadline > frame.index.max():
            continue  # 样本末尾没走完的仓位不计入，否则等于用截断改善结果

        side = int(event.side)
        exit_ts, exit_price, reason = _exit_scan(
            frame, now, entry_price, side, deadline, stop_loss, take_profit
        )
        if reason == "no_data":
            continue
        gross = side * (exit_price / entry_price - 1.0)
        net = gross - entry_cost - exit_cost
        open_positions.append((exit_ts, event.inst_id))
        per_day[day] = per_day.get(day, 0) + 1
        trades.append({
            "event_id": getattr(event, "event_id", ""),
            "inst_id": event.inst_id,
            "event_type": event.event_type,
            "side": side,
            "entry_ts": now,
            "exit_ts": exit_ts,
            "hold_hours": (exit_ts - now).total_seconds() / 3600.0,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "gross": gross,
            "net": net,
            "pnl": net * rules.position_notional,
            "reason": reason,
            "event_day": event.event_day,
            "signal": getattr(event, "abs_signal", np.nan),
        })

    frame = pd.DataFrame(trades)
    return Result(frame, summarize_trades(frame, rules), equity_curve(frame))


def equity_curve(trades: pd.DataFrame) -> pd.Series:
    if trades.empty:
        return pd.Series(dtype=float)
    ordered = trades.sort_values("exit_ts")
    return pd.Series(ordered.pnl.cumsum().to_numpy(), index=ordered.exit_ts)


def summarize_trades(trades: pd.DataFrame, rules: Rules) -> dict:
    """回测结论。核心是胜率、单笔期望和出手频率，不是总收益。

    总收益在六周样本上没有意义，可以靠加杠杆随意放大；胜率和
    单笔净期望才是能外推的东西。
    """
    if trades.empty:
        return {"n_trades": 0}
    net = trades["net"]
    wins, losses = net[net > 0], net[net <= 0]
    equity = equity_curve(trades)
    peak = equity.cummax()
    drawdown = (equity - peak)
    span_days = max(
        (trades.exit_ts.max() - trades.entry_ts.min()).total_seconds() / 86400.0, 1.0
    )
    daily = trades.groupby(trades.entry_ts.dt.date)["net"].mean()
    return {
        "n_trades": int(len(trades)),
        "n_days_traded": int(trades.entry_ts.dt.date.nunique()),
        "trades_per_week": float(len(trades) / span_days * 7),
        "win_rate": float((net > 0).mean()),
        "avg_net_bps": float(net.mean() * 1e4),
        "median_net_bps": float(net.median() * 1e4),
        "avg_win_bps": float(wins.mean() * 1e4) if len(wins) else 0.0,
        "avg_loss_bps": float(losses.mean() * 1e4) if len(losses) else 0.0,
        "profit_factor": float(wins.sum() / abs(losses.sum())) if len(losses) and losses.sum() != 0 else np.inf,
        "expectancy_bps": float(net.mean() * 1e4),
        "total_pnl": float(trades.pnl.sum()),
        "max_drawdown": float(drawdown.min()) if len(drawdown) else 0.0,
        # 按日均收益算的年化夏普，仅用于横向比较不同规则，不做绝对解读。
        "sharpe_daily": float(daily.mean() / daily.std() * np.sqrt(252)) if daily.std() > 0 else np.nan,
        "avg_hold_hours": float(trades.hold_hours.mean()),
        "stop_rate": float((trades.reason == "stop").mean()),
        "short_share": float((trades.side < 0).mean()),
    }
