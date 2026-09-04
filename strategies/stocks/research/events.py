"""事件检测器。

每个检测器输出一张事件表，字段统一：

    inst_id      标的
    event_ts     决策时刻（此刻之前的信息全部已知，此刻可以成交）
    event_type   事件类型
    side         +1 = 利多（做多）, -1 = 利空（做空/回避）
    ...          该事件特有的特征列

铁律：event_ts 之后的任何数据都不能参与检测。K 线在加载时已经
右移到收盘时间，所以"用到 event_ts 当根 bar 的收盘价"是合法的。

当前只保留一个可进入回测和实盘的事件：休市期间相对正股收盘锚点的
错位，方向按反转处理，并在正股开盘附近退出。其他事件假设已从项目中移除，
避免把未验证方向重新带回生产链路。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..market import sessions
from ..config import Config

MAX_STALE = pd.Timedelta(hours=2)


def price_asof(frame: pd.DataFrame, ts: pd.Timestamp,
               max_stale: pd.Timedelta = MAX_STALE) -> float | None:
    """ts 时刻最新的已知收盘价；超过 max_stale 没有成交则视为未知。

    深夜的代币化股票经常整根 bar 无成交。把上一次成交价当作当前
    可成交价会系统性低估休市错位，所以宁可把这个事件丢掉。
    """
    index = frame.index
    position = index.searchsorted(ts, side="right") - 1
    if position < 0:
        return None
    stamp = index[position]
    if ts - stamp > max_stale:
        return None
    value = frame["close"].iloc[position]
    return float(value) if np.isfinite(value) else None


def _window_slice(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return frame.loc[(frame.index > start) & (frame.index <= end)]


_CASH_RETURN_CACHE: dict[tuple[int, int], pd.Series] = {}


def cash_session_returns(frame: pd.DataFrame) -> pd.Series:
    """只在美股现货时段内的对数收益，用来估 beta。

    休市时段的代币收益里混着大量流动性噪音，拿它估出来的 beta 会
    被噪音压低（衰减偏差）。现货时段代币紧跟底层，beta 才有意义。

    结果按 (对象身份, 长度) 缓存：beta 要在每个事件的时点上重估，
    而 market_state 在全历史索引上并不便宜，不缓存会让参数敏感性
    扫描慢上两个数量级。
    """
    key = (id(frame), len(frame))
    cached = _CASH_RETURN_CACHE.get(key)
    if cached is not None:
        return cached
    state = sessions.market_state(frame.index)
    returns = np.log(frame["close"]).diff()
    # 跨越休市缺口的那一根收益不属于现货时段内部，必须剔除。
    contiguous = state.eq("cash") & state.shift(1).eq("cash")
    result = returns.where(contiguous).dropna()
    _CASH_RETURN_CACHE[key] = result
    return result


def estimate_beta(
    target: pd.DataFrame, driver: pd.DataFrame, until: pd.Timestamp,
    min_obs: int = 80, default: float = 1.0, cap: float = 4.0,
) -> float:
    """用 until 之前的现货时段收益估 beta，样本不足就退回默认值。

    严格只用 until 之前的数据：beta 是事件检测的一部分，用全样本
    估出来的 beta 会把未来信息带进事件定义。
    """
    # 先在全历史上算好收益（可缓存），再按 until 截断。反过来做会让
    # 每个时点都重算一次 market_state，缓存也就失效了。
    a = cash_session_returns(target)
    b = cash_session_returns(driver)
    a = a.loc[a.index <= until]
    b = b.loc[b.index <= until]
    joined = pd.concat([a.rename("y"), b.rename("x")], axis=1).dropna()
    if len(joined) < min_obs or joined["x"].var() == 0:
        return default
    beta = joined["y"].cov(joined["x"]) / joined["x"].var()
    if not np.isfinite(beta):
        return default
    return float(np.clip(beta, -cap, cap))


def _trailing_offhours_volume(frame: pd.DataFrame, until: pd.Timestamp,
                              lookback_days: float = 10.0) -> float:
    """休市时段的常态成交额中位数，作为放量的比较基准。

    必须只跟休市时段比：现货时段的成交额高一个数量级，拿它当基准
    会让任何盘后放量都显得微不足道。
    """
    start = until - pd.Timedelta(days=lookback_days)
    recent = frame.loc[(frame.index > start) & (frame.index <= until)]
    if recent.empty:
        return float("nan")
    state = sessions.market_state(recent.index)
    off = recent.loc[state.eq("closed").to_numpy(), "volume_quote"]
    if len(off) < 20:
        return float("nan")
    return float(off.median())


# --------------------------------------------------------------------------
# 事件一：休市错位
# --------------------------------------------------------------------------

def off_hours_dislocation(
    frames: dict[str, pd.DataFrame], config: Config,
    benchmark: pd.DataFrame | None = None,
    windows: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """底层休市期间，代币相对现货收盘锚点的偏离。

    每个休市窗口每个标的最多触发一次（首次越过阈值），这样一段
    持续的漂移不会被切成几十个互相重叠的"独立"事件。
    """
    if not frames:
        return pd.DataFrame()
    span_start = min(f.index.min() for f in frames.values())
    span_end = max(f.index.max() for f in frames.values())
    if windows is None:
        windows = sessions.closed_windows(span_start, span_end)
    windows = windows.loc[windows.hours >= config.min_closed_hours]

    threshold = config.dislocation_bps / 1e4
    rows = []
    for _, window in windows.iterrows():
        close_ts, open_ts = window.close_ts, window.open_ts
        market_dev = np.nan
        if benchmark is not None:
            anchor_m = price_asof(benchmark, close_ts)
            last_m = price_asof(benchmark, open_ts)
            if anchor_m and last_m:
                market_dev = last_m / anchor_m - 1.0

        for inst_id, frame in frames.items():
            anchor = price_asof(frame, close_ts)
            if anchor is None or anchor <= 0:
                continue
            inside = _window_slice(frame, close_ts, open_ts)
            if inside.empty:
                continue
            deviation = inside["close"] / anchor - 1.0
            breached = deviation.abs() >= threshold
            if not breached.any():
                continue
            event_ts = deviation.index[breached.argmax()]
            hours_left = (open_ts - event_ts).total_seconds() / 3600.0
            # 离开盘太近就没有回归的空间，成本也吃不掉。
            if hours_left < 1.0:
                continue
            dev = float(deviation.loc[event_ts])
            baseline = _trailing_offhours_volume(frame, close_ts)
            burst = _window_slice(frame, close_ts, event_ts)["volume_quote"]
            volume_ratio = (
                float(burst.median() / baseline)
                if np.isfinite(baseline) and baseline > 0 and len(burst)
                else np.nan
            )
            beta = (
                estimate_beta(frame, benchmark, close_ts) if benchmark is not None else 1.0
            )
            market_move = np.nan
            if benchmark is not None:
                anchor_m = price_asof(benchmark, close_ts)
                now_m = price_asof(benchmark, event_ts)
                if anchor_m and now_m:
                    market_move = now_m / anchor_m - 1.0
            residual = dev - beta * market_move if np.isfinite(market_move) else np.nan

            rows.append({
                "inst_id": inst_id,
                "event_ts": event_ts,
                "event_type": "off_hours_dislocation",
                # 回归假设：偏离多少，就往反方向下注多少。
                "side": -1 if dev > 0 else 1,
                "deviation": dev,
                "residual_deviation": residual,
                "market_deviation": market_dev,
                "beta": beta,
                "volume_ratio": volume_ratio,
                "hours_to_open": hours_left,
                "window_kind": window.kind,
                "window_hours": window.hours,
                "anchor_close_ts": close_ts,
                "resolve_ts": open_ts,
                "event_day": pd.Timestamp(window.next_day),
            })
    return pd.DataFrame(rows)

def attach_trailing_liquidity(
    events: pd.DataFrame, frames: dict[str, pd.DataFrame], bar: str = "15m",
) -> pd.DataFrame:
    """给每个事件挂上事件时刻之前 24 小时的真实成交额。

    用今天的 24h 成交额筛标的会把选择偏差带进历史：一个八月才放量
    的标的，在七月其实根本不可交易。滚动成交额是逐事件的、只看
    过去的流动性判据，也是回测里唯一诚实的可交易性证据。
    """
    if events.empty:
        return events
    bars_per_day = int(pd.Timedelta(days=1) / pd.Timedelta(bar.replace("H", "h")))
    cache: dict[str, pd.Series] = {}
    for inst_id, frame in frames.items():
        cache[inst_id] = frame["volume_quote"].rolling(bars_per_day, min_periods=bars_per_day // 2).sum()
    values = []
    for event in events.itertuples(index=False):
        series = cache.get(event.inst_id)
        if series is None:
            values.append(np.nan)
            continue
        position = series.index.searchsorted(event.event_ts, side="right") - 1
        values.append(float(series.iloc[position]) if position >= 0 else np.nan)
    out = events.copy()
    out["trailing_quote_volume_24h"] = values
    return out
