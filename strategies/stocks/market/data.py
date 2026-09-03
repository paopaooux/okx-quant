"""K 线缓存的读写与面板拼装。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import Config
from .okx import OKXClient, OKXError, update_cache


def refresh(
    client: OKXClient, inst_ids: list[str], config: Config, bars: list[str] | None = None,
    verbose: bool = True,
) -> dict[str, list[str]]:
    """把所有标的的 K 线拉到本地。返回每个 bar 上成功的标的列表。"""
    bars = bars or [config.bar]
    done: dict[str, list[str]] = {}
    for bar in bars:
        ok: list[str] = []
        for index, inst_id in enumerate(inst_ids, 1):
            try:
                update_cache(client, inst_id, bar, config.history_days, config.data_dir)
                ok.append(inst_id)
            except OKXError as exc:
                if verbose:
                    print(f"  [{bar}] {inst_id} 跳过: {exc}")
                continue
            if verbose and index % 10 == 0:
                print(f"  [{bar}] {index}/{len(inst_ids)}")
        done[bar] = ok
        if verbose:
            print(f"[{bar}] 缓存完成 {len(ok)}/{len(inst_ids)}")
    return done


def load_frame(inst_id: str, bar: str, data_dir: str | Path) -> pd.DataFrame | None:
    path = Path(data_dir) / bar / f"{inst_id}.csv"
    if not path.exists():
        return None
    frame = pd.read_csv(path, parse_dates=["ts"])
    if frame.empty:
        return None
    frame["ts"] = pd.to_datetime(frame["ts"], utc=True)
    return frame.sort_values("ts").drop_duplicates("ts").reset_index(drop=True)


def load_panel(
    inst_ids: list[str], bar: str, data_dir: str | Path,
) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for inst_id in inst_ids:
        frame = load_frame(inst_id, bar, data_dir)
        if frame is not None and len(frame) > 50:
            frames[inst_id] = frame.set_index("ts")
    return frames


def close_matrix(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """所有标的的收盘价对齐成宽表，缺失不前向填充。

    前向填充会凭空造出"停牌期间价格没变"的假象，而代币化股票在
    深夜确实会出现整根 bar 没有成交的情况——那时候真实可成交价是
    未知的，把它当成已知会让休市错位被系统性低估。
    """
    if not frames:
        return pd.DataFrame()
    return pd.DataFrame({inst: frame["close"] for inst, frame in frames.items()}).sort_index()


def coverage(frames: dict[str, pd.DataFrame], bar: str) -> pd.DataFrame:
    """每个标的的样本覆盖情况，用来判断哪些标的还不够格进研究。"""
    step = pd.Timedelta(bar.replace("H", "h"))
    rows = []
    for inst_id, frame in frames.items():
        span = frame.index.max() - frame.index.min()
        expected = max(span / step, 1)
        rows.append({
            "instId": inst_id,
            "bars": len(frame),
            "start": frame.index.min(),
            "end": frame.index.max(),
            "days": span.total_seconds() / 86400.0,
            "fill_ratio": len(frame) / expected,
            "median_quote_vol": frame["volume_quote"].median(),
        })
    return pd.DataFrame(rows).sort_values("days", ascending=False).reset_index(drop=True)


BAR_DURATION = {
    "1m": pd.Timedelta(minutes=1), "3m": pd.Timedelta(minutes=3),
    "5m": pd.Timedelta(minutes=5), "15m": pd.Timedelta(minutes=15),
    "30m": pd.Timedelta(minutes=30), "1H": pd.Timedelta(hours=1),
    "4H": pd.Timedelta(hours=4), "1D": pd.Timedelta(days=1),
}


def to_bar_end(frames: dict[str, pd.DataFrame], bar: str) -> dict[str, pd.DataFrame]:
    """把索引从 bar 的开盘时间改成收盘时间。

    OKX 返回的时间戳是 bar 的起点，而收盘价要到 bar 结束才成立。
    如果直接拿起点时间当决策时间，就等于用未来 15 分钟的信息做当下
    的判断——这类前视偏差在事件研究里足以凭空造出全部"收益"。
    整个项目统一在加载时右移一个 bar 长度，之后所有时间戳都是
    "这一刻确实已经知道的信息"。
    """
    delta = BAR_DURATION[bar]
    return {
        inst: frame.set_index(frame.index + delta) for inst, frame in frames.items()
    }
