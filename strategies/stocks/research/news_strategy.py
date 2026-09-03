"""公告事件 -> 利多/利空 -> 吃一波行情。

时序是这套东西的全部。三个时刻必须严格分开：

    accepted            8-K 被 SEC 受理，信息在这一秒变成公开信息
    accepted + observe  观察窗结束，**这里才允许下单**
    next_open + hold    正股开盘之后离场

方向不是猜的。item 代码那条路走不通：窗口内 1.01/2.01 一共 3 笔、
3.02 一类 6 笔，方向命中率 33% 和 50%，样本量决定了它永远不会有
统计意义。真正可用的方向来源是**代币自己在 observe 窗口里的走势**
——公告落在美股休市时段（窗口内 141/141 都是），正股不能动，只有
代币能交易，所以这 30 分钟的价格变化就是市场对"利多还是利空"投的
票。策略赌的是这一票的方向在正股开盘时会被延续而不是被推翻。
实测回归斜率 +0.227：延续。

`observe` 里的价格只用 <= accepted+observe 的 bar，入场价取
observe 窗结束后第一根可成交 bar 的收盘价。data.to_bar_end() 已经把
索引右移成收盘时刻，所以"索引 <= 决策时刻"就等于"这一刻确实已知"。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..market import sessions

# 只有这些表格带方向信息。10-K/10-Q 是定期报告，市场早知道日期；
# 8-K 是"发生了一件需要立刻告知投资者的事"，是真正的事件。
EVENT_FORMS = ("8-K", "8-K/A")


def _price_at(frame: pd.DataFrame, ts: pd.Timestamp, max_stale: pd.Timedelta) -> float | None:
    """ts 时刻最后一个已成交的收盘价，太旧就当作没有。"""
    position = frame.index.searchsorted(ts, side="right") - 1
    if position < 0:
        return None
    stamp = frame.index[position]
    if ts - stamp > max_stale:
        return None
    price = float(frame["close"].iloc[position])
    return price if np.isfinite(price) and price > 0 else None


def build_events(
    filings: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
    observe_minutes: float = 30.0,
    hold_minutes: float = 120.0,
    min_move_bps: float = 0.0,
    max_stale_minutes: float = 90.0,
    require_closed: bool = True,
    time_shift_days: float = 0.0,
) -> pd.DataFrame:
    """把申报记录变成可回测的事件表。

    `time_shift_days` 是安慰剂开关：把事件时间整体平移若干天，其余
    一切照旧。真实机制在 0 上应该显著、在任何非零平移上应该消失。
    这是整套研究里唯一真正能否证"是不是我在拟合噪声"的检验。
    """
    observe = pd.Timedelta(minutes=observe_minutes)
    max_stale = pd.Timedelta(minutes=max_stale_minutes)
    shift = pd.Timedelta(days=time_shift_days)

    subset = filings.loc[filings.form.isin(EVENT_FORMS)].copy()
    rows: list[dict] = []
    for record in subset.itertuples():
        frame = frames.get(record.instId)
        if frame is None:
            continue
        accepted = pd.Timestamp(record.accepted) + shift
        decision = accepted + observe
        if require_closed and sessions.market_state(pd.DatetimeIndex([accepted])).iloc[0] != "closed":
            continue
        # 观察窗与入场都必须完整落在样本内，否则等于用截断改结果。
        if not (frame.index.min() <= accepted and decision <= frame.index.max()):
            continue

        anchor = _price_at(frame, accepted, max_stale)
        decided = _price_at(frame, decision, max_stale)
        if anchor is None or decided is None:
            continue
        move = decided / anchor - 1.0
        # move 恰好为 0 说明观察窗里一根成交都没有，是"没有信息"而不是
        # "看空"，必须丢掉而不是按 move > 0 的三目静默归到空头。
        if move == 0.0 or abs(move) * 1e4 < min_move_bps:
            continue

        entry_position = frame.index.searchsorted(decision, side="right") - 1
        if entry_position < 0:
            continue
        resolve = sessions.next_open(decision)
        if resolve is None:
            continue
        rows.append({
            "inst_id": record.instId,
            "ticker": record.ticker,
            "event_type": "sec_8k_momentum",
            "event_ts": frame.index[entry_position],
            "accepted": accepted,
            "resolve_ts": resolve,
            "event_day": pd.Timestamp(decision.date()),
            "side": 1 if move > 0 else -1,
            "observe_move": move,
            # 排队时按观察窗涨跌幅取前几名。这是逐标的的量，同一时刻
            # 不同标的不会打平（对比 btc_leadlag 用板块共同变量排序时
            # 名额全被同一个标的拿走的退化）。
            "abs_move_bps": abs(move) * 1e4,
            "items": record.items if isinstance(record.items, str) else "",
        })

    events = pd.DataFrame(rows)
    if events.empty:
        return events
    return events.sort_values("event_ts").reset_index(drop=True)


def annotate_shortable(events: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    """标出哪些空头信号其实下不了单。

    OKX 的代币化股票只有一部分开了杠杆/借币（lever > 0），其余只能
    现货做多。不做这一步的回测会把大量不可执行的空单算成收益。
    """
    lever = universe.set_index("instId").lever.astype(float)
    events = events.copy()
    events["lever"] = events.inst_id.map(lever).fillna(0.0)
    events["executable"] = (events.side > 0) | (events.lever > 0)
    return events


def build_agent_events(
    filings: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
    verdicts: dict[str, dict],
    latency_minutes: float = 5.0,
    min_conviction: int = 1,
    max_stale_minutes: float = 90.0,
    time_shift_days: float = 0.0,
) -> pd.DataFrame:
    """方向来自 agent 读原文，不来自价格。

    和 `build_events` 的关键差别是**入场时刻提前了 25 分钟**。原来那 30
    分钟观察窗是纯成本：它的唯一作用是让价格把方向"告诉"我，代价是这
    段行情已经走完了。agent 在受理那一秒就有方向，所以只需要留出真实
    的处理延迟——实测单条判定 19 秒，抓原文几秒，留 5 分钟是宽松的。

    这也意味着这条线不再有循环味道：方向来自文本，价格只在结算时出现
    一次。`latency_minutes` 之后的第一根 bar 收盘价是入场价，而
    `data.to_bar_end()` 已经把索引右移成收盘时刻，所以"索引 <= 入场时刻"
    就等于"这个价格确实已经发生过"。
    """
    latency = pd.Timedelta(minutes=latency_minutes)
    max_stale = pd.Timedelta(minutes=max_stale_minutes)
    shift = pd.Timedelta(days=time_shift_days)

    subset = filings.loc[filings.form.isin(EVENT_FORMS)].copy()
    rows: list[dict] = []
    for record in subset.itertuples():
        verdict = verdicts.get(record.accession)
        if not verdict or "direction" not in verdict:
            continue
        direction = verdict["direction"]
        conviction = int(verdict.get("conviction", 1))
        # neutral 是 agent 明确说"这条不该动"，不是缺失值。它必须被丢掉
        # 而不是硬塞一个方向——"少出手"的第一道闸门就在这里。
        if direction == "neutral" or conviction < min_conviction:
            continue

        frame = frames.get(record.instId)
        if frame is None:
            continue
        accepted = pd.Timestamp(record.accepted) + shift
        entry_at = accepted + latency
        if not (frame.index.min() <= accepted and entry_at <= frame.index.max()):
            continue
        # 必须取**结束时刻 >= 入场时刻**的第一根 bar。用"结束 <= 入场"的
        # 最后一根是错的：公告 :07 落地、入场 :12，那样会拿到 :00 收盘的
        # 价格——新闻还没公开时的价，等于免费拿到整段事件行情。这个方向
        # 的错误只会虚增收益，所以宁可保守一根。
        position = int(frame.index.searchsorted(entry_at, side="left"))
        if position >= len(frame.index):
            continue
        entry_ts = frame.index[position]
        # bar 结束得太晚说明中间断了行情，这笔的入场价不可信。
        if entry_ts - entry_at > max_stale:
            continue
        entry_price = float(frame["close"].iloc[position])
        if not np.isfinite(entry_price) or entry_price <= 0:
            continue
        resolve = sessions.next_open(entry_ts)
        if resolve is None:
            continue

        rows.append({
            "inst_id": record.instId,
            "ticker": record.ticker,
            "event_type": "sec_8k_agent",
            "event_ts": entry_ts,
            "accepted": accepted,
            "resolve_ts": resolve,
            "event_day": pd.Timestamp(entry_at.date()),
            "side": 1 if direction == "bullish" else -1,
            "agent_direction": direction,
            "conviction": conviction,
            "category": verdict.get("category", "other"),
            "reason": verdict.get("reason", ""),
            # 排队用 conviction。它是逐标的、逐公告的量，同一时刻不会打平，
            # 而且和价格完全无关——这正是之前一直缺的那个独立排序变量。
            "abs_move_bps": float(conviction),
            "lead_hours": float((resolve - accepted).total_seconds() / 3600.0),
            "market_state": sessions.market_state(pd.DatetimeIndex([accepted])).iloc[0],
            "items": record.items if isinstance(record.items, str) else "",
            "accession": record.accession,
        })

    events = pd.DataFrame(rows)
    if events.empty:
        return events
    return events.sort_values("event_ts").reset_index(drop=True)
