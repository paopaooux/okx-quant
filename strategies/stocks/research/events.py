"""事件检测器。

每个检测器输出一张事件表，字段统一：

    inst_id      标的
    event_ts     决策时刻（此刻之前的信息全部已知，此刻可以成交）
    event_type   事件类型
    side         +1 = 利多（做多）, -1 = 利空（做空/回避）
    ...          该事件特有的特征列

铁律：event_ts 之后的任何数据都不能参与检测。K 线在加载时已经
右移到收盘时间，所以"用到 event_ts 当根 bar 的收盘价"是合法的。

事件被分成两族，对应用户要回答的那个问题——这波是利多还是利空：

  信息族：休市期间真有消息（财报、宏观、加密价格），代币的价格
          变动是对新信息的定价，方向应当延续。
  噪音族：休市期间没有消息，只是薄盘口被少量订单推动，代币偏离了
          底层的真实价值，方向应当在下一次现货开盘时回归。

放量与否是区分两族最直接的观测量，也是本项目的核心假设。
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


# --------------------------------------------------------------------------
# 事件二：盘后放量冲击（财报型）
# --------------------------------------------------------------------------

def after_hours_shock(
    frames: dict[str, pd.DataFrame], config: Config,
    hours_after_close: float = 3.0,
    windows: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """现货收盘后数小时内的放量急动。

    美股财报绝大多数在 16:00 ET 收盘后立刻发布，所以这个时间窗
    天然富集真实信息事件，不需要外部财报日历也能把它们捞出来。
    放量是"有人真的知道点什么"的可观测代理。
    """
    if not frames:
        return pd.DataFrame()
    span_start = min(f.index.min() for f in frames.values())
    span_end = max(f.index.max() for f in frames.values())
    if windows is None:
        windows = sessions.closed_windows(span_start, span_end)

    threshold = config.dislocation_bps / 1e4
    rows = []
    for _, window in windows.iterrows():
        close_ts, open_ts = window.close_ts, window.open_ts
        deadline = min(close_ts + pd.Timedelta(hours=hours_after_close), open_ts)
        for inst_id, frame in frames.items():
            anchor = price_asof(frame, close_ts)
            if anchor is None or anchor <= 0:
                continue
            inside = _window_slice(frame, close_ts, deadline)
            if inside.empty:
                continue
            baseline = _trailing_offhours_volume(frame, close_ts)
            if not np.isfinite(baseline) or baseline <= 0:
                continue
            move = inside["close"] / anchor - 1.0
            ratio = inside["volume_quote"] / baseline
            qualified = (move.abs() >= threshold) & (ratio >= config.volume_shock_multiple)
            if not qualified.any():
                continue
            event_ts = qualified.index[qualified.argmax()]
            shock = float(move.loc[event_ts])
            rows.append({
                "inst_id": inst_id,
                "event_ts": event_ts,
                "event_type": "after_hours_shock",
                # 信息假设：真消息带来的重定价应当延续。
                "side": 1 if shock > 0 else -1,
                "shock": shock,
                "volume_ratio": float(ratio.loc[event_ts]),
                "hours_after_close": (event_ts - close_ts).total_seconds() / 3600.0,
                "hours_to_open": (open_ts - event_ts).total_seconds() / 3600.0,
                "window_kind": window.kind,
                "anchor_close_ts": close_ts,
                "resolve_ts": open_ts,
                "event_day": pd.Timestamp(window.next_day),
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 事件三：加密财库股与 BTC 的联动缺口
# --------------------------------------------------------------------------

def crypto_beta_gap(
    frames: dict[str, pd.DataFrame], driver: pd.DataFrame, config: Config,
    windows: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """BTC 在美股休市期间动了，而加密财库股代币还没跟上。

    MSTR / COIN / STRC 这类公司的股价对 BTC 有稳定且很高的 beta。
    BTC 是 24/7 的，美股不是——所以每天有十几个小时，BTC 已经把
    新信息定价了，而代币化股票还挂在昨天的收盘价上。这个缺口是
    这个市场里为数不多的、方向可以事先说清楚的事件。
    """
    if not frames:
        return pd.DataFrame()
    span_start = min(f.index.min() for f in frames.values())
    span_end = max(f.index.max() for f in frames.values())
    if windows is None:
        windows = sessions.closed_windows(span_start, span_end)
    windows = windows.loc[windows.hours >= config.min_closed_hours]

    threshold = config.crypto_beta_gap_bps / 1e4
    rows = []
    for _, window in windows.iterrows():
        close_ts, open_ts = window.close_ts, window.open_ts
        driver_anchor = price_asof(driver, close_ts)
        if driver_anchor is None or driver_anchor <= 0:
            continue
        for inst_id, frame in frames.items():
            anchor = price_asof(frame, close_ts)
            if anchor is None or anchor <= 0:
                continue
            beta = estimate_beta(frame, driver, close_ts, default=np.nan)
            if not np.isfinite(beta) or beta <= 0.2:
                # beta 太低说明这个标的其实不跟 BTC，缺口没有含义。
                continue
            inside = _window_slice(frame, close_ts, open_ts)
            if inside.empty:
                continue
            driver_inside = driver.reindex(inside.index).ffill(limit=4)
            driver_move = driver_inside["close"] / driver_anchor - 1.0
            token_move = inside["close"] / anchor - 1.0
            gap = beta * driver_move - token_move
            breached = gap.abs() >= threshold
            if not breached.fillna(False).any():
                continue
            event_ts = gap.index[breached.fillna(False).argmax()]
            hours_left = (open_ts - event_ts).total_seconds() / 3600.0
            if hours_left < 1.0:
                continue
            rows.append({
                "inst_id": inst_id,
                "event_ts": event_ts,
                "event_type": "crypto_beta_gap",
                # 追赶假设：代币应当朝 BTC 已经指出的方向补上。
                "side": 1 if gap.loc[event_ts] > 0 else -1,
                "gap": float(gap.loc[event_ts]),
                "driver_move": float(driver_move.loc[event_ts]),
                "token_move": float(token_move.loc[event_ts]),
                "beta": beta,
                "hours_to_open": hours_left,
                "window_kind": window.kind,
                "anchor_close_ts": close_ts,
                "resolve_ts": open_ts,
                "event_day": pd.Timestamp(window.next_day),
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 事件四：上市首日
# --------------------------------------------------------------------------

def listing_debut(
    frames: dict[str, pd.DataFrame], listings: pd.DataFrame,
    delay_hours: float = 2.0,
) -> pd.DataFrame:
    """OKX 上新代币化股票后固定延迟的那一刻。

    延迟是必要的：开盘前几根 bar 的价格由集合竞价和极薄的盘口决定，
    既不可成交也不可复现。
    """
    rows = []
    lookup = listings.set_index("instId")["list_ts"].to_dict()
    for inst_id, frame in frames.items():
        list_ts = lookup.get(inst_id)
        if list_ts is None:
            continue
        event_ts_target = pd.Timestamp(list_ts) + pd.Timedelta(hours=delay_hours)
        after = frame.loc[frame.index >= event_ts_target]
        if after.empty:
            continue
        event_ts = after.index[0]
        first = price_asof(frame, pd.Timestamp(list_ts) + pd.Timedelta(minutes=30))
        now = float(after["close"].iloc[0])
        rows.append({
            "inst_id": inst_id,
            "event_ts": event_ts,
            "event_type": "listing_debut",
            # 首发溢价衰减假设，方向由研究结果证实或推翻。
            "side": -1,
            "debut_move": (now / first - 1.0) if first else np.nan,
            "list_ts": pd.Timestamp(list_ts),
            "event_day": pd.Timestamp(event_ts.date()),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 事件五：开盘跳空
# --------------------------------------------------------------------------

def open_gap(
    frames: dict[str, pd.DataFrame], config: Config,
    windows: pd.DataFrame | None = None, settle_minutes: float = 30.0,
) -> pd.DataFrame:
    """现货开盘后代币完成收敛，形成相对昨收的跳空。

    与前几个事件不同，这个事件发生在底层交易的时段内，代币此刻
    有真实价格锚。它回答的是另一个问题：真实跳空之后是延续还是
    回补——也就是休市错位交易该在哪一刻离场。
    """
    if not frames:
        return pd.DataFrame()
    span_start = min(f.index.min() for f in frames.values())
    span_end = max(f.index.max() for f in frames.values())
    if windows is None:
        windows = sessions.closed_windows(span_start, span_end)

    threshold = config.dislocation_bps / 1e4
    rows = []
    for _, window in windows.iterrows():
        close_ts, open_ts = window.close_ts, window.open_ts
        event_ts_target = open_ts + pd.Timedelta(minutes=settle_minutes)
        for inst_id, frame in frames.items():
            anchor = price_asof(frame, close_ts)
            settled = price_asof(frame, event_ts_target, max_stale=pd.Timedelta(minutes=60))
            if not anchor or not settled:
                continue
            gap = settled / anchor - 1.0
            if abs(gap) < threshold:
                continue
            after = frame.loc[frame.index >= event_ts_target]
            if after.empty:
                continue
            rows.append({
                "inst_id": inst_id,
                "event_ts": after.index[0],
                "event_type": "open_gap",
                "side": 1 if gap > 0 else -1,
                "gap": float(gap),
                "window_kind": window.kind,
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
