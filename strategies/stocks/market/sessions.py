"""美股现货交易时段日历（UTC 口径）。

代币化美股的所有事件定义都挂在这个日历上：底层股票开着的时候
代币有真实价格锚，关着的时候没有。夏令时会让同一个 ET 时刻在
UTC 上漂移一小时，所以必须用真正的时区库而不是固定偏移。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

EASTERN = ZoneInfo("America/New_York")
REGULAR_OPEN = time(9, 30)
REGULAR_CLOSE = time(16, 0)
EARLY_CLOSE = time(13, 0)

# NYSE 休市日。覆盖代币化美股可能存在的全部区间，宁可多列几年。
HOLIDAYS: frozenset[date] = frozenset(
    date(*parts) for parts in [
        (2025, 1, 1), (2025, 1, 20), (2025, 2, 17), (2025, 4, 18), (2025, 5, 26),
        (2025, 6, 19), (2025, 7, 4), (2025, 9, 1), (2025, 11, 27), (2025, 12, 25),
        (2026, 1, 1), (2026, 1, 19), (2026, 2, 16), (2026, 4, 3), (2026, 5, 25),
        (2026, 6, 19), (2026, 7, 3), (2026, 9, 7), (2026, 11, 26), (2026, 12, 25),
        (2027, 1, 1), (2027, 1, 18), (2027, 2, 15), (2027, 3, 26), (2027, 5, 31),
        (2027, 6, 18), (2027, 7, 5), (2027, 9, 6), (2027, 11, 25), (2027, 12, 24),
    ]
)

# 13:00 ET 提前收盘日。
EARLY_CLOSES: frozenset[date] = frozenset(
    date(*parts) for parts in [
        (2025, 7, 3), (2025, 11, 28), (2025, 12, 24),
        (2026, 11, 27), (2026, 12, 24),
        (2027, 11, 26),
    ]
)


def is_trading_day(day: date) -> bool:
    return day.weekday() < 5 and day not in HOLIDAYS


@dataclass(frozen=True)
class Session:
    day: date
    open_ts: pd.Timestamp   # UTC
    close_ts: pd.Timestamp  # UTC
    early: bool


def session_for(day: date) -> Session | None:
    """某个自然日的现货时段，非交易日返回 None。"""
    if not is_trading_day(day):
        return None
    early = day in EARLY_CLOSES
    close_local = EARLY_CLOSE if early else REGULAR_CLOSE
    open_utc = datetime.combine(day, REGULAR_OPEN, tzinfo=EASTERN).astimezone(ZoneInfo("UTC"))
    close_utc = datetime.combine(day, close_local, tzinfo=EASTERN).astimezone(ZoneInfo("UTC"))
    return Session(day, pd.Timestamp(open_utc), pd.Timestamp(close_utc), early)


def sessions_between(start: pd.Timestamp, end: pd.Timestamp) -> list[Session]:
    """[start, end] 覆盖到的全部现货时段，按时间排序。

    两端各放宽一天，保证区间边界上的那个时段不会被截掉。
    """
    start = pd.Timestamp(start).tz_convert("UTC")
    end = pd.Timestamp(end).tz_convert("UTC")
    first = (start.tz_convert(EASTERN) - timedelta(days=1)).date()
    last = (end.tz_convert(EASTERN) + timedelta(days=1)).date()
    out: list[Session] = []
    day = first
    while day <= last:
        session = session_for(day)
        if session is not None:
            out.append(session)
        day += timedelta(days=1)
    return out


@dataclass(frozen=True)
class ClosedWindow:
    """一段底层股票停止交易的区间：上一次收盘 -> 下一次开盘。"""

    close_ts: pd.Timestamp     # 锚点：底层最后一个真实价格的时刻
    open_ts: pd.Timestamp      # 收敛点：底层下一次给出真实价格的时刻
    prev_day: date
    next_day: date
    hours: float
    kind: str                  # overnight / weekend / holiday


def closed_windows(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """区间内所有休市窗口。

    kind 按跨越的日历天数分：隔夜约 17.5 小时，周末约 65 小时，
    夹着假日的更长。窗口越长，代币失锚越久，错位越大——这是
    事件强度的天然分组变量，不是需要拟合的参数。
    """
    sessions = sessions_between(start, end)
    rows = []
    for prev, nxt in zip(sessions, sessions[1:]):
        gap_days = (nxt.day - prev.day).days
        if gap_days >= 3:
            kind = "holiday" if nxt.day.weekday() != 0 or gap_days > 3 else "weekend"
        elif gap_days == 3 and nxt.day.weekday() == 0:
            kind = "weekend"
        elif gap_days == 1:
            kind = "overnight"
        else:
            kind = "holiday"
        rows.append({
            "close_ts": prev.close_ts,
            "open_ts": nxt.open_ts,
            "prev_day": prev.day,
            "next_day": nxt.day,
            "hours": (nxt.open_ts - prev.close_ts).total_seconds() / 3600.0,
            "kind": kind,
        })
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.loc[(frame.open_ts > start) & (frame.close_ts < end)].reset_index(drop=True)


def market_state(index: pd.DatetimeIndex) -> pd.Series:
    """给每个 UTC 时间戳标记 cash / closed。

    向量化实现：先把时间戳换到 ET，再用当地日期和当地时间判断。
    循环版本在 93 个标的 × 数千根 bar 上会明显拖慢研究脚本。
    """
    index = pd.DatetimeIndex(index)
    local = index.tz_convert(EASTERN)
    days = local.normalize()
    minutes = local.hour * 60 + local.minute
    holiday = np.isin(days.date, np.array(sorted(HOLIDAYS), dtype=object))
    weekday = local.weekday < 5
    early = np.isin(days.date, np.array(sorted(EARLY_CLOSES), dtype=object))
    close_minute = np.where(early, EARLY_CLOSE.hour * 60, REGULAR_CLOSE.hour * 60)
    open_minute = REGULAR_OPEN.hour * 60 + REGULAR_OPEN.minute
    open_now = weekday & ~holiday & (minutes >= open_minute) & (minutes < close_minute)
    return pd.Series(np.where(open_now, "cash", "closed"), index=index, name="state")


def previous_close(ts: pd.Timestamp) -> pd.Timestamp | None:
    """ts 之前最近一次现货收盘时刻。"""
    ts = pd.Timestamp(ts).tz_convert("UTC")
    day = ts.tz_convert(EASTERN).date()
    for _ in range(12):
        session = session_for(day)
        if session is not None and session.close_ts <= ts:
            return session.close_ts
        day -= timedelta(days=1)
    return None


def next_open(ts: pd.Timestamp) -> pd.Timestamp | None:
    """ts 之后最近一次现货开盘时刻。"""
    ts = pd.Timestamp(ts).tz_convert("UTC")
    day = ts.tz_convert(EASTERN).date()
    for _ in range(12):
        session = session_for(day)
        if session is not None and session.open_ts > ts:
            return session.open_ts
        day += timedelta(days=1)
    return None
