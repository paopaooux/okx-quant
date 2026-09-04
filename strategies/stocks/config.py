"""研究配置。

默认值刻意偏保守：这套数据只有六周，任何靠调参调出来的漂亮结果
都不可信，所以门槛（最少事件数、最低胜率、成本假设）设在
"过不去就别做"的位置。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os

try:
    import yaml
except ModuleNotFoundError:  # YAML is only needed for custom config files.
    yaml = None


@dataclass
class Config:
    # ---- 数据 ----
    bar: str = "15m"
    fine_bar: str = "5m"          # 开盘/收盘附近需要更细的粒度
    history_days: int = 120
    data_dir: str = field(default_factory=lambda: os.environ.get(
        "STOCK_ALPHA_DATA_DIR",
        str(Path(os.environ["STOCK_ALPHA_ROOT"]) / "data") if os.environ.get("STOCK_ALPHA_ROOT") else "data/stocks_swap",
    ))
    result_dir: str = field(default_factory=lambda: os.environ.get(
        "STOCK_ALPHA_RESULT_DIR",
        str(Path(os.environ["STOCK_ALPHA_ROOT"]) / "results") if os.environ.get("STOCK_ALPHA_ROOT") else "results/stocks_offhours_research",
    ))

    # ---- 标的宇宙 ----
    min_quote_volume_24h: float = 150_000   # 24h 计价成交额下限
    max_universe: int = 60
    min_history_days: float = 7.0

    # ---- 成本 ----
    # OKX 股票永续的合约费率与滑点；盘口薄，滑点按保守口径估计。
    fee_bps: float = 10.0
    slippage_bps: float = 12.0

    # ---- 事件阈值 ----
    dislocation_bps: float = 150.0          # 休市错位触发线
    min_closed_hours: float = 8.0           # 太短的休市窗口不构成事件

    # ---- 加密财库 / 加密相关股票 ----
    crypto_proxies: list[str] = field(default_factory=lambda: [
        "XMSTR", "XCOIN", "XSTRC", "XBMNR", "XIREN", "XCRCL", "XHOOD",
        "XGME", "XBE", "XSHAZ", "XRIVN",
    ])
    @classmethod
    def load(cls, path: str | Path | None) -> "Config":
        if path is None:
            return cls()
        if yaml is None:
            raise RuntimeError("PyYAML is required to load a stock strategy YAML config")
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        allowed = set(cls.__dataclass_fields__)
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"未知配置项: {', '.join(sorted(unknown))}")
        return cls(**raw)

    @property
    def round_trip_bps(self) -> float:
        """一次完整进出的成本，进出各一次手续费加滑点。"""
        return 2.0 * (self.fee_bps + self.slippage_bps)
