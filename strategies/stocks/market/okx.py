"""OKX 公共行情客户端。

只用公共端点：代币化美股的研究不需要私钥，也不应该在研究阶段
持有下单能力。TLS/UA 的处理沿用已经在生产里验证过的做法。
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import os
import re
import socket
import ssl
import time

import pandas as pd
import requests
from requests.adapters import HTTPAdapter


class OKXError(RuntimeError):
    pass


class _TLS12Adapter(HTTPAdapter):
    """Work around upstream proxy paths that terminate OKX TLS 1.3 handshakes."""

    def init_poolmanager(self, *args, **kwargs):
        context = ssl.create_default_context()
        context.maximum_version = ssl.TLSVersion.TLSv1_2
        kwargs["ssl_context"] = context
        return super().init_poolmanager(*args, **kwargs)

    def proxy_manager_for(self, proxy, **proxy_kwargs):
        context = ssl.create_default_context()
        context.maximum_version = ssl.TLSVersion.TLSv1_2
        proxy_kwargs["ssl_context"] = context
        return super().proxy_manager_for(proxy, **proxy_kwargs)


def _local_clash_proxy() -> str | None:
    """发现 clashctl/mihomo 的本地 HTTP(mixed) 端口。

    本机环境通常只设置 CLASHCTL_HOME，并不会导出 HTTPS_PROXY；此前
    `trust_env=False` 因此让 OKX 直连并触发内部 DNS。只读取配置中的
    mixed-port/port，并确认端口正在监听，不会自动启动或修改 Clash。
    """
    root = os.environ.get("CLASHCTL_HOME")
    if not root:
        return None
    candidates = [os.path.join(root, "resources", name)
                  for name in ("runtime.yaml", "config.yaml", "mixin.yaml")]
    ports: list[int] = []
    for path in candidates:
        try:
            text = Path(path).read_text(encoding="utf-8")
        except OSError:
            continue
        for key in ("mixed-port", "port"):
            match = re.search(rf"^\s*{re.escape(key)}\s*:\s*(\d+)\s*$", text, re.M)
            if match:
                ports.append(int(match.group(1)))
    # Prefer mixed-port. If only a plain HTTP port is configured it is also
    # compatible with requests' HTTP CONNECT proxy handling.
    for port in dict.fromkeys(ports):
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.3):
                return f"http://127.0.0.1:{port}"
        except OSError:
            continue
    return None


class OKXClient:
    def __init__(
        self, base_url: str = "https://www.okx.com", timeout: int = 20,
        proxy_url: str | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.trust_env = False
        self.session.mount("https://", _TLS12Adapter())
        # OKX 的边缘节点会在请求到达 API 之前用 403 拒绝程序化 UA，
        # 所以即使是公共行情端点也必须带浏览器 UA。
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                          " (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
            "Accept": "application/json",
        })
        proxy_url = (
            proxy_url
            or os.environ.get("OKX_PROXY_URL")
            or os.environ.get("okx_proxy_url")
            or os.environ.get("https_proxy")
            or os.environ.get("HTTPS_PROXY")
            or os.environ.get("http_proxy")
            or os.environ.get("HTTP_PROXY")
            or _local_clash_proxy()
        )
        if proxy_url:
            self.session.proxies.update({"http": proxy_url, "https": proxy_url})
        self.proxy_url = proxy_url

    def _get(self, path: str, params: dict | None = None) -> list[dict]:
        last_error: Exception | None = None
        for attempt in range(4):
            try:
                response = self.session.get(
                    self.base_url + path, params=params, timeout=self.timeout
                )
                response.raise_for_status()
                payload = response.json()
                if payload.get("code") != "0":
                    raise OKXError(f"OKX {payload.get('code')}: {payload.get('msg')}")
                return payload["data"]
            except (requests.RequestException, ValueError, OKXError) as exc:
                last_error = exc
                if attempt < 3:
                    time.sleep(1.5 * (attempt + 1))
        raise OKXError(f"请求 OKX 失败: {last_error}")

    def spot_instruments(self) -> pd.DataFrame:
        """所有现货交易对的静态信息，含 instCategory 与上市时间。

        instCategory == "3" 是 OKX 给代币化股票的分类，这是唯一可靠的
        识别方式：按 X 前缀猜会把 XRP / XLM / XAUT / XCH 这些真实加密
        资产错误地拉进来。
        """
        rows = self._get("/api/v5/public/instruments", {"instType": "SPOT"})
        frame = pd.DataFrame(rows)
        keep = [
            "instId", "baseCcy", "quoteCcy", "state", "instCategory",
            "listTime", "lever", "minSz", "lotSz", "tickSz", "maxMktAmt",
        ]
        frame = frame.loc[:, [c for c in keep if c in frame.columns]].copy()
        frame["list_ts"] = pd.to_datetime(
            pd.to_numeric(frame["listTime"], errors="coerce"), unit="ms", utc=True
        )
        for col in ("minSz", "lotSz", "tickSz", "maxMktAmt"):
            if col in frame.columns:
                frame[col] = pd.to_numeric(frame[col], errors="coerce")
        return frame.drop(columns=["listTime"])

    def spot_tickers(self) -> pd.DataFrame:
        rows = self._get("/api/v5/market/tickers", {"instType": "SPOT"})
        frame = pd.DataFrame(rows)
        out = pd.DataFrame({"instId": frame["instId"]})
        out["last"] = pd.to_numeric(frame["last"], errors="coerce")
        # volCcy24h 对现货是以计价货币计的成交额，正是我们要的美元口径。
        out["quote_volume_24h"] = pd.to_numeric(frame["volCcy24h"], errors="coerce")
        return out.dropna()

    def history_candles(
        self, inst_id: str, bar: str, start_ms: int, max_pages: int = 200,
    ) -> pd.DataFrame:
        """完整历史 K 线，只保留已收盘的 bar。

        history-candles 每页 300 根且只能向更早翻页，所以 max_pages 就是
        真实的历史上限；代币化美股上市不到两个月，默认页数远够。
        """
        records: dict[int, list] = {}
        after: int | None = None
        for _ in range(max_pages):
            params = {"instId": inst_id, "bar": bar, "limit": "300"}
            if after is not None:
                params["after"] = str(after)
            batch = self._get("/api/v5/market/history-candles", params)
            if not batch:
                break
            for row in batch:
                if row[8] == "1":  # 只要已确认收盘的 bar
                    records[int(row[0])] = row
            oldest = min(int(row[0]) for row in batch)
            if oldest <= start_ms or after == oldest:
                break
            after = oldest
            time.sleep(0.12)
        if not records:
            raise OKXError(f"{inst_id} 没有获取到 K 线")
        columns = ["ts", "open", "high", "low", "close", "volume",
                   "volume_ccy", "volume_quote", "confirm"]
        frame = pd.DataFrame(list(records.values()), columns=columns)
        frame["ts"] = pd.to_datetime(frame["ts"].astype("int64"), unit="ms", utc=True)
        for col in ["open", "high", "low", "close", "volume", "volume_quote"]:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        floor = pd.to_datetime(start_ms, unit="ms", utc=True)
        return (
            frame.loc[frame["ts"] >= floor,
                      ["ts", "open", "high", "low", "close", "volume", "volume_quote"]]
            .sort_values("ts")
            .dropna()
            .reset_index(drop=True)
        )


def update_cache(
    client: OKXClient, inst_id: str, bar: str, days: int, data_dir: str | Path,
    max_pages: int = 200,
) -> Path:
    """把 K 线落到 data_dir/bar/inst_id.csv，增量刷新尾部。"""
    path = Path(data_dir) / bar / f"{inst_id}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    requested_start = int((datetime.now(timezone.utc).timestamp() - days * 86400) * 1000)
    bar_ms = {"1m": 60_000, "3m": 180_000, "5m": 300_000,
              "15m": 900_000, "30m": 1_800_000, "1H": 3_600_000}.get(bar, 300_000)
    start_ms = requested_start
    old = None
    if path.exists():
        old = pd.read_csv(path, parse_dates=["ts"])
        if len(old):
            old_start_ms = int(pd.to_datetime(old.ts.min(), utc=True).timestamp() * 1000)
            old_end_ms = int(pd.to_datetime(old.ts.max(), utc=True).timestamp() * 1000)
            if requested_start >= old_start_ms:
                # 已经覆盖到要求的起点，只请求尾部并重叠最后一根已确认
                # K 线。history-candles 只保留 confirm=1，因此不会引入
                # 未完成 bar；重叠用于修复交易所刚确认的最新 bar。
                start_ms = max(requested_start, old_end_ms - bar_ms)
    fresh = client.history_candles(inst_id, bar, start_ms, max_pages)
    if old is not None:
        fresh = pd.concat([old, fresh], ignore_index=True)
    fresh["ts"] = pd.to_datetime(fresh["ts"], utc=True)
    fresh = fresh.drop_duplicates("ts", keep="last").sort_values("ts")
    # Keep the configured rolling window bounded even when the process has
    # been offline longer than `days`.
    floor = pd.to_datetime(requested_start, unit="ms", utc=True)
    fresh = fresh.loc[fresh["ts"] >= floor].reset_index(drop=True)
    # 研究脚本可能与刷新并发运行，先写完整的兄弟文件再原子替换。
    temporary = path.with_suffix(path.suffix + ".tmp")
    fresh.to_csv(temporary, index=False)
    temporary.replace(path)
    return path
