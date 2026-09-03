"""代币化美股的标的宇宙。

识别只认 OKX 自己的 instCategory == "3"。按 X 前缀猜会把 XRP、XLM、
XAUT、XCH、XTZ 这些真实加密资产错拉进来，而漏掉不带 X 的品种；
交易所的分类字段是唯一不需要维护的判据。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import Config
from .okx import OKXClient

TOKENIZED_STOCK_CATEGORY = "3"

# instCategory 只说明"是股票"，不说明是哪一类。指数 ETF 和杠杆 ETF
# 的行为跟单一个股不同，必须能分开统计。
INDEX_TOKENS = {"XSPY", "XQQQ", "XIWM", "XEWY", "XXLE", "XSMH", "XTTWO"}
LEVERAGED_TOKENS = {"XSOXL", "XTQQQ", "XSPCX", "XLRCX", "XBX", "XLITE"}


def discover(client: OKXClient, config: Config) -> pd.DataFrame:
    """当前可交易的代币化美股，含上市时间与 24 小时成交额。"""
    instruments = client.spot_instruments()
    stocks = instruments.loc[
        (instruments.instCategory == TOKENIZED_STOCK_CATEGORY)
        & (instruments.quoteCcy == "USDT")
        & (instruments.state == "live")
    ].copy()
    if stocks.empty:
        raise RuntimeError("OKX 没有返回任何代币化股票；检查网络或 instCategory 定义是否变更")
    tickers = client.spot_tickers()
    merged = stocks.merge(tickers, on="instId", how="left")
    merged["quote_volume_24h"] = merged["quote_volume_24h"].fillna(0.0)
    merged["base"] = merged["baseCcy"]
    merged["kind"] = "single"
    merged.loc[merged.base.isin(INDEX_TOKENS), "kind"] = "index"
    merged.loc[merged.base.isin(LEVERAGED_TOKENS), "kind"] = "leveraged"
    merged["crypto_proxy"] = merged.base.isin(set(config.crypto_proxies))
    merged["margin"] = merged["lever"].replace("", "0").astype(float) > 0
    return merged.sort_values("quote_volume_24h", ascending=False).reset_index(drop=True)


def tradable(frame: pd.DataFrame, config: Config) -> pd.DataFrame:
    """加上流动性与历史长度过滤后的研究宇宙。

    杠杆 ETF 代币被排除：它们每日重置，隔夜收益不是标的的隔夜收益，
    休市错位回归的逻辑在它们身上不成立。
    """
    now = pd.Timestamp.utcnow()
    age_days = (now - frame.list_ts).dt.total_seconds() / 86400.0
    keep = (
        (frame.quote_volume_24h >= config.min_quote_volume_24h)
        & (age_days >= config.min_history_days)
        & (frame.kind != "leveraged")
    )
    return frame.loc[keep].head(config.max_universe).reset_index(drop=True)


def save(frame: pd.DataFrame, result_dir: str | Path) -> Path:
    path = Path(result_dir) / "universe.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path
