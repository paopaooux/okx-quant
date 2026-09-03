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
        str(Path(os.environ["STOCK_ALPHA_ROOT"]) / "data") if os.environ.get("STOCK_ALPHA_ROOT") else "data/stocks",
    ))
    result_dir: str = field(default_factory=lambda: os.environ.get(
        "STOCK_ALPHA_RESULT_DIR",
        str(Path(os.environ["STOCK_ALPHA_ROOT"]) / "results") if os.environ.get("STOCK_ALPHA_ROOT") else "results/stocks",
    ))

    # ---- 标的宇宙 ----
    min_quote_volume_24h: float = 150_000   # 24h 计价成交额下限
    max_universe: int = 60
    listing_burn_in_hours: float = 48.0     # 上市初期定价混乱，事件研究里剔除
    min_history_days: float = 7.0

    # ---- 成本 ----
    # OKX 现货吃单费约 10bp；代币化股票盘口薄，滑点按两倍于普通币种估。
    fee_bps: float = 10.0
    slippage_bps: float = 12.0
    stress_slippage_bps: float = 30.0

    # ---- 事件阈值 ----
    dislocation_bps: float = 150.0          # 休市错位触发线
    volume_shock_multiple: float = 3.0      # 盘后放量倍数，用来区分信息与噪音
    crypto_beta_gap_bps: float = 200.0      # 加密财库股与 BTC 的偏离触发线
    min_closed_hours: float = 8.0           # 太短的休市窗口不构成事件

    # ---- 出手与晋级门槛（少出手 / 高胜率）----
    min_events: int = 40                    # 池化事件数下限
    min_event_days: int = 15                # 独立交易日下限，防止一天的行情撑起全部样本
    min_win_rate: float = 0.60
    min_net_edge_bps: float = 25.0          # 扣完成本后的平均收益下限
    max_p_value: float = 0.05
    min_half_sample_agreement: bool = True  # 前后半段必须同号
    max_trades_per_day: int = 3             # 少出手

    # ---- 加密财库 / 加密相关股票 ----
    crypto_proxies: list[str] = field(default_factory=lambda: [
        "XMSTR", "XCOIN", "XSTRC", "XBMNR", "XIREN", "XCRCL", "XHOOD",
        "XGME", "XBE", "XSHAZ", "XRIVN",
    ])
    benchmark_inst: str = "XSPY-USDT"
    crypto_benchmark_inst: str = "BTC-USDT"

    # ---- 互联网科技情报（只登记事件，不直接产生交易信号）----
    # GitHub 是模型发布最稳定的官方机器可读来源；30 分钟轮询可在未配置
    # token 的 60 requests/hour 限额内运行。配置 GITHUB_TOKEN 后限额更高。
    internet_poll_minutes: float = 30.0
    # 首次同步回看一周，后续仍按 source_id 去重，只记录新增事件。
    internet_lookback_hours: float = 168.0
    internet_github_orgs: list[str] = field(default_factory=lambda: [
        "zai-org", "THUDM", "deepseek-ai", "QwenLM", "MoonshotAI",
        "MiniMax-AI", "openai", "google-deepmind", "meta-llama", "mistralai",
    ])
    internet_rss_feeds: list[str] = field(default_factory=lambda: [
        "https://openai.com/news/rss.xml",
        "https://blog.google/technology/ai/rss/",
        "https://blogs.nvidia.com/feed/",
        # 可靠媒体往往早于官方确认模型评测、产品泄露和供应链消息。
        "https://techcrunch.com/category/artificial-intelligence/feed/",
        "https://www.theverge.com/rss/ai-artificial-intelligence/index.xml",
        "https://www.technologyreview.com/feed/",
        "https://feeds.arstechnica.com/arstechnica/technology-lab",
        # 研究与工程源：模型论文、训练/部署实践通常早于公司公告。
        "https://arxiv.org/rss/cs.AI",
        "https://aws.amazon.com/blogs/machine-learning/feed/",
        "https://www.amazon.science/index.rss",
    ])
    internet_watch_pages: list[str] = field(default_factory=lambda: [
        "https://docs.bigmodel.cn/cn/guide/start/model-overview",
    ])
    # 公开社区信号；默认只抓少量高相关查询，避免把泛科技噪声灌给 LLM。
    internet_hackernews_queries: list[str] = field(default_factory=lambda: [
        "large language model", "AI benchmark", "GLM model",
    ])
    internet_reddit_subreddits: list[str] = field(default_factory=lambda: [
        "MachineLearning", "LocalLLaMA", "wallstreetbets",
    ])

    # ---- 舆情 LLM（与 SEC agent 独立配置）----
    # codex_cli 使用本机 Codex 登录态，不依赖 OPENAI_API_KEY。model 留空时
    # 沿用 Codex CLI 当前配置；需要固定型号时在 YAML 里显式填写。
    sentiment_llm_provider: str = "codex_cli"
    sentiment_llm_model: str = ""
    sentiment_llm_timeout_seconds: int = 60
    sentiment_llm_batch_size: int = 6
    sentiment_llm_cache_dir: str = "data/sentiment_verdicts"
    sentiment_latency_minutes: float = 15.0
    sentiment_min_materiality: int = 3
    sentiment_min_surprise: int = 2
    sentiment_min_confidence: int = 3

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
