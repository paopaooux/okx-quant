"""通过检验的信号：BTC 领先加密财库股。

前面的研究否掉了三个假设（休市错位回归、盘后冲击延续、开盘跳空
延续），只有一个留了下来，机制也是这批假设里唯一有明确因果方向的：

    BTC 24 小时交易，美股不是。当 BTC 在美股休市期间大幅移动，
    资产负债表里装着比特币的公司（MSTR / CRCL / BMNR / COIN）
    的真实价值已经变了，但它的股票要等到现货开盘才能重新定价。
    代币化股票让这个等待期变成可交易的。

三条支持证据，重要性从高到低：

  1. 安慰剂检验：把 BTC 收益按天平移 ±1 / ±2 / 3 天，相关性全部
     消失，只在真实时间对齐时存在（Spearman 0.456, p=0.011）。
  2. 对照组：同样高 beta 但业务与加密无关的半导体股票代币，
     方向命中率 43%；低 beta 组 50%。效应是加密特有的。
  3. 方向命中率 70%（30 个休市窗口中 21 个）。

以及一条必须记住的限制：开盘之后的 4 小时里，这个关系会反转
（t = -1.79）。收益全部集中在开盘那一瞬间的收敛上，拿进盘中会吐回去。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import events as ev
from ..market import sessions
from ..config import Config

# 加密财库/加密业务公司。名单按业务性质而不是按统计 beta 定：
# 半导体股票的 BTC beta 也接近 1，但那来自整体风险偏好，
# 开盘时没有需要补的加密定价缺口。
CRYPTO_NATIVE = [
    "XMSTR-USDT", "XCRCL-USDT", "XBMNR-USDT", "XCOIN-USDT",
    "XIREN-USDT", "XHOOD-USDT", "XSTRC-USDT", "XBE-USDT",
]

# OKX 上支持杠杆（因而可以做空）的代币化股票。其余只能做多，
# 这不是选择而是交易所的限制，必须体现在回测里。
SHORTABLE = {"XMSTR-USDT", "XCRCL-USDT"}


def btc_leadlag_events(
    stock_frames: dict[str, pd.DataFrame],
    btc: pd.DataFrame,
    config: Config,
    threshold_bps: float = 150.0,
    lead_hours: float = 2.0,
    min_beta: float = 0.8,
    cohort: list[str] | None = None,
    allow_short: bool = True,
    windows: pd.DataFrame | None = None,
    min_trailing_volume: float | None = None,
) -> pd.DataFrame:
    """生成事件表。

    lead_hours 是开盘前的观测提前量：既要让 BTC 的移动已经形成，
    又要留出真的能挂单成交的时间。取 0 会得到一个漂亮但不可交易的
    回测——那一刻的价格只有开盘之后才知道。
    """
    cohort = cohort or CRYPTO_NATIVE
    min_volume = (
        config.min_quote_volume_24h if min_trailing_volume is None else min_trailing_volume
    )
    frames = {i: f for i, f in stock_frames.items() if i in set(cohort)}
    if not frames or btc is None:
        return pd.DataFrame()
    if windows is None:
        windows = sessions.closed_windows(
            min(f.index.min() for f in frames.values()),
            max(f.index.max() for f in frames.values()),
        )
    windows = windows.loc[windows.hours >= config.min_closed_hours]

    threshold = threshold_bps / 1e4
    bars_per_day = int(pd.Timedelta(days=1) / pd.Timedelta(config.bar.replace("H", "h")))
    volume_24h = {
        inst: frame["volume_quote"].rolling(bars_per_day, min_periods=bars_per_day // 2).sum()
        for inst, frame in frames.items()
    }

    rows = []
    for _, window in windows.iterrows():
        close_ts, open_ts = window.close_ts, window.open_ts
        observe_ts = open_ts - pd.Timedelta(hours=lead_hours)
        if observe_ts <= close_ts:
            continue
        btc_anchor = ev.price_asof(btc, close_ts)
        btc_now = ev.price_asof(btc, observe_ts)
        if not btc_anchor or not btc_now:
            continue
        btc_move = btc_now / btc_anchor - 1.0
        if abs(btc_move) < threshold:
            continue
        side = 1 if btc_move > 0 else -1

        for inst_id, frame in frames.items():
            if side < 0 and (not allow_short or inst_id not in SHORTABLE):
                continue
            # beta 只用观测时刻之前的数据估，事件定义里不能有未来信息。
            beta = ev.estimate_beta(frame, btc, observe_ts, default=np.nan)
            if not np.isfinite(beta) or beta < min_beta:
                continue
            series = volume_24h[inst_id]
            position = series.index.searchsorted(observe_ts, side="right") - 1
            if position < 0:
                continue
            trailing = float(series.iloc[position])
            if not np.isfinite(trailing) or trailing < min_volume:
                continue
            entry = ev.price_asof(frame, observe_ts, pd.Timedelta(hours=1))
            if not entry:
                continue
            token_move = entry / (ev.price_asof(frame, close_ts) or np.nan) - 1.0
            rows.append({
                "inst_id": inst_id,
                "event_ts": frame.index[frame.index.searchsorted(observe_ts, side="right") - 1],
                "event_type": "btc_leadlag",
                "side": side,
                "btc_move": btc_move,
                "token_move": token_move,
                "unabsorbed": beta * btc_move - token_move,
                "beta": beta,
                "window_kind": window.kind,
                "window_hours": window.hours,
                "resolve_ts": open_ts,
                "anchor_close_ts": close_ts,
                "trailing_quote_volume_24h": trailing,
                "event_day": pd.Timestamp(window.next_day),
            })
    return pd.DataFrame(rows)
