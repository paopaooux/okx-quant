"""把逐笔交易变成组合层面的账。

`summarize_trades` 报的是逐笔的东西（胜率、单笔期望），那是判断"信号
有没有用"的尺子。换手率、回撤、年化是另一回事——它们只有在有资金约
束的组合上才有定义：同一时刻最多几个仓、每个仓占多少本金、亏了之后
下一笔是不是变小。这里补上这一层。

一个必须说在前面的限制：**六周样本的年化基本没有意义**。样本里独立
交易日约 22 天，年化等于把噪声乘以 sqrt(252/22) ≈ 3.4 倍（收益本身
乘 11 倍）。它照要求算出来，但读的时候要当成"如果这个日均能维持"的
外推，不是预测。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _recovery_stats(curve: pd.Series) -> dict:
    """Return drawdown recovery durations from the realized equity curve.

    The curve is event/exit based, so these figures intentionally describe
    realized-equity recovery and do not claim to include intrabar floating PnL.
    """
    if curve.empty:
        return {"max_drawdown_recovery_days": np.nan,
                "longest_drawdown_recovery_days": np.nan,
                "longest_drawdown_days": np.nan,
                "unrecovered_drawdown": False}
    values = curve.sort_index().to_numpy(dtype=float)
    times = curve.sort_index().index
    peaks = np.maximum.accumulate(values)
    active = None
    episodes = []
    for i, value in enumerate(values):
        if active is None and value < peaks[i] * (1 - 1e-12):
            active = {
                "peak_ts": times[i - 1] if i else times[i],
                "peak_value": peaks[i],
                "trough_ts": times[i],
                "trough_value": value,
            }
        elif active is not None:
            if value < active["trough_value"]:
                active["trough_ts"] = times[i]
                active["trough_value"] = value
            if value >= active["peak_value"] - 1e-8:
                episodes.append({
                    "depth": active["trough_value"] / active["peak_value"] - 1.0,
                    "recovery_days": (times[i] - active["trough_ts"]).total_seconds() / 86400.0,
                    "duration_days": (times[i] - active["peak_ts"]).total_seconds() / 86400.0,
                })
                active = None
    if active is not None:
        deepest = min(episodes, key=lambda e: e["depth"], default=None)
        return {
            "max_drawdown_recovery_days": float(deepest["recovery_days"]) if deepest else np.nan,
            "longest_drawdown_recovery_days": float(max((e["recovery_days"] for e in episodes), default=np.nan)),
            "longest_drawdown_days": float((times[-1] - active["peak_ts"]).total_seconds() / 86400.0),
            "unrecovered_drawdown": True,
        }
    longest = float(max((e["recovery_days"] for e in episodes), default=np.nan))
    deepest = min(episodes, key=lambda e: e["depth"], default=None)
    return {
        "max_drawdown_recovery_days": float(deepest["recovery_days"]) if deepest else np.nan,
        "longest_drawdown_recovery_days": longest,
        "longest_drawdown_days": float(max((e["duration_days"] for e in episodes), default=np.nan)),
        "unrecovered_drawdown": False,
    }


def simulate(
    trades: pd.DataFrame,
    capital: float = 100_000.0,
    max_concurrent: int = 3,
) -> tuple[pd.Series, dict]:
    """事件驱动地跑一遍资金曲线。

    每笔按**入场时权益**的 1/max_concurrent 下注，所以亏损之后仓位自动
    缩小（复利，不是固定手数）。权益只在离场时刻更新——这是收盘价口径
    的曲线，会**低估**盘中回撤，因为持仓期间的浮亏没有被记进去。
    """
    if trades.empty:
        return pd.Series(dtype=float), {"n_trades": 0}

    weight = 1.0 / max(max_concurrent, 1)
    ordered = trades.sort_values("entry_ts").reset_index(drop=True)

    # 入场看当时的权益，离场才结算，所以要按时间轴推进而不是按笔推进。
    events: list[tuple[pd.Timestamp, str, int]] = []
    for i, row in ordered.iterrows():
        events.append((row.entry_ts, "in", i))
        events.append((row.exit_ts, "out", i))
    events.sort(key=lambda e: (e[0], 0 if e[1] == "out" else 1))

    equity = capital
    notional = np.zeros(len(ordered))
    curve_ts: list[pd.Timestamp] = [ordered.entry_ts.min()]
    curve_eq: list[float] = [capital]
    traded_notional = 0.0
    exposure_hours = 0.0

    for stamp, kind, idx in events:
        if kind == "in":
            size = equity * weight
            notional[idx] = size
            traded_notional += size
            exposure_hours += float(ordered.hold_hours.iloc[idx]) * weight
        else:
            equity += notional[idx] * float(ordered.net.iloc[idx])
            traded_notional += notional[idx]  # 平仓也是一次成交
            curve_ts.append(stamp)
            curve_eq.append(equity)

    curve = pd.Series(curve_eq, index=pd.DatetimeIndex(curve_ts)).sort_index()
    peak = curve.cummax()
    drawdown = curve / peak - 1.0

    span_days = max(
        (ordered.exit_ts.max() - ordered.entry_ts.min()).total_seconds() / 86400.0, 1.0)
    total_return = equity / capital - 1.0
    years = span_days / 365.25
    annualized = (1.0 + total_return) ** (1.0 / years) - 1.0 if total_return > -1 else -1.0

    # 日收益用于夏普。样本只有二十几天，这个夏普的置信区间极宽。
    daily = curve.resample("1D").last().ffill().pct_change().dropna()

    recovery = _recovery_stats(curve)
    return curve, {
        "capital": capital,
        "final_equity": float(equity),
        "total_return": float(total_return),
        "annualized_return": float(annualized),
        "max_drawdown_pct": float(drawdown.min()),
        "max_drawdown_abs": float((curve - peak).min()),
        # 换手率 = 双边成交额 / 平均权益。1.0 表示整个区间把本金完整
        # 换了一次手（买+卖各算一次）。
        "turnover_total": float(traded_notional / curve.mean()),
        "turnover_annualized": float(traded_notional / curve.mean() * 365.25 / span_days),
        # 资金占用率：平均有多少比例的本金真的在市场里。少出手的策略
        # 这个数很低，意味着大部分时间是空仓的。
        "avg_exposure": float(exposure_hours / (span_days * 24.0)),
        "span_days": float(span_days),
        "n_trades": int(len(ordered)),
        "sharpe_daily": float(daily.mean() / daily.std() * np.sqrt(252)) if daily.std() > 0 else np.nan,
        "vol_annualized": float(daily.std() * np.sqrt(252)) if len(daily) > 1 else np.nan,
        **recovery,
    }
