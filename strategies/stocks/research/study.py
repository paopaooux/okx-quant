"""事件研究引擎。

组合回测在六周样本上必然过拟合：选股规则、权重、止损、调仓频率
互相纠缠，网格搜索无论有没有 alpha 都会吐出一个"最优格"。事件
研究把这些自由度全部拿掉——事件定义好之后，剩下的只有前向收益。

显著性一律按事件日聚类。同一天几十个代币化美股的休市错位高度
相关（都被同一个隔夜期指、同一个宏观数据推动），把它们当成独立
样本会把 t 值放大好几倍。真正的独立样本量是休市窗口的个数，
这套数据里只有 32 个。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from ..config import Config

# 休市错位类事件的自然持有期就是"到下一次现货开盘"，此外再看几个
# 固定时长，用来判断收益是不是集中在收敛那一刻。
HORIZONS: dict[str, pd.Timedelta] = {
    "1h": pd.Timedelta(hours=1),
    "4h": pd.Timedelta(hours=4),
    "12h": pd.Timedelta(hours=12),
    "1d": pd.Timedelta(days=1),
    "3d": pd.Timedelta(days=3),
}


def _return_between(frame: pd.DataFrame, start_ts: pd.Timestamp, end_ts: pd.Timestamp,
                    max_stale: pd.Timedelta = pd.Timedelta(hours=2)) -> float:
    """start_ts 收盘买入、end_ts 收盘卖出的简单收益。

    两端都要求确实有成交价，缺一端就返回 NaN 而不是拿相邻价格顶上：
    在薄盘代币上，用不存在的价格结算等于给自己发钱。
    """
    if end_ts > frame.index.max():
        return np.nan
    index = frame.index
    start_pos = index.searchsorted(start_ts, side="right") - 1
    end_pos = index.searchsorted(end_ts, side="right") - 1
    if start_pos < 0 or end_pos < 0 or end_pos <= start_pos:
        return np.nan
    if start_ts - index[start_pos] > max_stale or end_ts - index[end_pos] > max_stale:
        return np.nan
    entry = float(frame["close"].iloc[start_pos])
    exit_price = float(frame["close"].iloc[end_pos])
    if not (np.isfinite(entry) and np.isfinite(exit_price)) or entry <= 0:
        return np.nan
    return exit_price / entry - 1.0


def evaluate(
    events: pd.DataFrame, frames: dict[str, pd.DataFrame], config: Config,
    horizons: dict[str, pd.Timedelta] | None = None,
    resolve_offset_minutes: float = 30.0,
) -> pd.DataFrame:
    """给每个事件算出各持有期的方向化净收益。

    side 已经把"利多还是利空"编码进去了，所以 signed 收益为正就是
    这个事件的方向判断对了。net 再扣掉一次完整进出的成本。
    """
    if events.empty:
        return pd.DataFrame()
    horizons = horizons or HORIZONS
    cost = config.round_trip_bps / 1e4
    rows = []
    for event in events.itertuples(index=False):
        frame = frames.get(event.inst_id)
        if frame is None:
            continue
        base = {
            "inst_id": event.inst_id,
            "event_ts": event.event_ts,
            "event_type": event.event_type,
            "side": event.side,
            "event_day": event.event_day,
        }
        targets = {name: event.event_ts + delta for name, delta in horizons.items()}
        # 收敛点退出：错位类事件的正确离场时刻是底层重新给出真实
        # 价格之后，而不是某个固定时长之后。
        resolve_ts = getattr(event, "resolve_ts", None)
        if resolve_ts is not None and pd.notna(resolve_ts):
            resolve_exit = pd.Timestamp(resolve_ts) + pd.Timedelta(minutes=resolve_offset_minutes)
            if resolve_exit > event.event_ts:
                targets["to_open"] = resolve_exit
        for name, exit_ts in targets.items():
            gross = _return_between(frame, event.event_ts, exit_ts)
            if not np.isfinite(gross):
                continue
            signed = event.side * gross
            rows.append({
                **base,
                "horizon": name,
                "hold_hours": (exit_ts - event.event_ts).total_seconds() / 3600.0,
                "gross": signed,
                "net": signed - cost,
            })
    return pd.DataFrame(rows)


def _day_clustered_test(frame: pd.DataFrame, column: str) -> tuple[float, float, int]:
    """先在事件日内取均值，再对日均值做单样本 t 检验。

    这是这份研究里最重要的一行统计：不做这一步，93 个标的同一天的
    同向错位会被当成 93 个独立观测，t 值虚高到必然"显著"。
    """
    daily = frame.groupby("event_day")[column].mean().dropna()
    if len(daily) < 3:
        return np.nan, np.nan, len(daily)
    result = stats.ttest_1samp(daily.to_numpy(), 0.0)
    return float(result.statistic), float(result.pvalue), len(daily)


def summarize(
    evaluated: pd.DataFrame, config: Config, by: list[str] | None = None,
) -> pd.DataFrame:
    """按事件类型 × 持有期（可再加分组维度）汇总。"""
    if evaluated.empty:
        return pd.DataFrame()
    keys = ["event_type", "horizon"] + (by or [])
    rows = []
    for values, group in evaluated.groupby(keys, dropna=False):
        values = values if isinstance(values, tuple) else (values,)
        # 检验必须做在净收益上。毛收益的 t 检验看不到一次进出 44bp 的
        # 成本，会让"毛收益显著为正但扣完成本不显著"的组合混过门槛——
        # 40bp/12h 那一格就是这样：毛 p=0.040，净 p=0.153。
        t_stat, p_value, n_days = _day_clustered_test(group, "net")
        t_gross, p_gross, _ = _day_clustered_test(group, "gross")
        naive = stats.ttest_1samp(group["net"].to_numpy(), 0.0) if len(group) > 2 else None
        # 前后半段一致性：把事件按时间切两半，两边符号必须一致。
        ordered = group.sort_values("event_ts")
        half = len(ordered) // 2
        first_half = ordered["net"].iloc[:half].mean() if half >= 3 else np.nan
        second_half = ordered["net"].iloc[half:].mean() if half >= 3 else np.nan
        rows.append({
            **dict(zip(keys, values)),
            "n_events": len(group),
            "n_days": n_days,
            "n_insts": group["inst_id"].nunique(),
            "gross_bps": group["gross"].mean() * 1e4,
            "net_bps": group["net"].mean() * 1e4,
            "median_net_bps": group["net"].median() * 1e4,
            "win_rate": float((group["net"] > 0).mean()),
            "gross_win_rate": float((group["gross"] > 0).mean()),
            "t_clustered": t_stat,
            "p_clustered": p_value,
            "t_clustered_gross": t_gross,
            "p_clustered_gross": p_gross,
            "t_naive": float(naive.statistic) if naive is not None else np.nan,
            "vol_bps": group["gross"].std() * 1e4,
            "first_half_net_bps": first_half * 1e4 if np.isfinite(first_half) else np.nan,
            "second_half_net_bps": second_half * 1e4 if np.isfinite(second_half) else np.nan,
            "halves_agree": bool(
                np.isfinite(first_half) and np.isfinite(second_half)
                and np.sign(first_half) == np.sign(second_half) and first_half > 0
            ),
        })
    frame = pd.DataFrame(rows)
    return frame.sort_values(["event_type", "net_bps"], ascending=[True, False]).reset_index(drop=True)


def gate(summary: pd.DataFrame, config: Config) -> pd.DataFrame:
    """晋级门槛：只有全部通过的组合才允许进入回测和实盘监听。

    门槛互相独立地卡住不同的失败方式：事件数和事件日卡样本量，
    胜率和净收益卡经济意义，聚类 p 值卡运气，前后半段一致性卡
    "全部收益来自某一周的一次行情"。
    """
    if summary.empty:
        return summary
    frame = summary.copy()
    frame["pass_events"] = frame.n_events >= config.min_events
    frame["pass_days"] = frame.n_days >= config.min_event_days
    frame["pass_win"] = frame.win_rate >= config.min_win_rate
    frame["pass_edge"] = frame.net_bps >= config.min_net_edge_bps
    frame["pass_pvalue"] = frame.p_clustered <= config.max_p_value
    frame["pass_stability"] = frame.halves_agree | (not config.min_half_sample_agreement)
    checks = ["pass_events", "pass_days", "pass_win", "pass_edge", "pass_pvalue", "pass_stability"]
    frame["promoted"] = frame[checks].all(axis=1)
    frame["failed_checks"] = frame[checks].apply(
        lambda row: ",".join(c.removeprefix("pass_") for c in checks if not row[c]), axis=1
    )
    return frame


def bucket(evaluated: pd.DataFrame, events: pd.DataFrame, column: str,
           quantiles: int = 3, labels: list[str] | None = None) -> pd.DataFrame:
    """把某个事件特征分位分组，挂回评估表。

    这是回答"利多还是利空"的主要工具：例如按放量倍数分组，看
    高放量的错位是否延续（信息）、低放量的是否回归（噪音）。
    """
    if evaluated.empty or events.empty or column not in events.columns:
        return evaluated
    keys = ["inst_id", "event_ts", "event_type"]
    feature = events[keys + [column]].drop_duplicates(keys)
    merged = evaluated.merge(feature, on=keys, how="left")
    values = merged[column]
    if values.notna().sum() < quantiles * 5:
        merged[f"{column}_bucket"] = "n/a"
        return merged
    try:
        merged[f"{column}_bucket"] = pd.qcut(
            values, quantiles, labels=labels or [f"q{i+1}" for i in range(quantiles)],
            duplicates="drop",
        ).astype(object)
    except ValueError:
        merged[f"{column}_bucket"] = "n/a"
    merged[f"{column}_bucket"] = merged[f"{column}_bucket"].fillna("n/a")
    return merged
