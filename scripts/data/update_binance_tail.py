"""Append the newest confirmed Binance Futures bars to the backtest archive.

Binance Vision daily zip files are published with a delay.  This updater uses
the public Futures REST endpoints for the recent tail, then writes the result
back into the same Binance files used by training/backtests.  It never feeds
these rows to the OKX live trader, so venue mixing cannot happen at runtime.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("BINANCE_DATA_DIR", ROOT / "data"))
STATUS = Path(os.environ.get("BINANCE_TAIL_STATUS", DATA / "binance_tail_status.json"))
BASE = os.environ.get("BINANCE_FAPI_URL", "https://fapi.binance.com")
SYMBOLS = (
    "ADAUSDT", "BNBUSDT", "BTCUSDT", "DOGEUSDT",
    "ETHUSDT", "LINKUSDT", "SOLUSDT", "XRPUSDT",
)


def _dotenv_value(name: str) -> str | None:
    """Read one optional local .env value without overriding the shell."""
    path = ROOT / ".env"
    try:
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if line.startswith(f"{name}="):
                value = line.split("=", 1)[1].strip()
                if value[:1] == value[-1:] and value[:1] in {"'", '"'}:
                    value = value[1:-1]
                return value or None
    except OSError:
        pass
    return None


def session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": "okx-quant/binance-tail"})
    proxy = (os.environ.get("BINANCE_PROXY_URL") or os.environ.get("HTTPS_PROXY")
             or os.environ.get("HTTP_PROXY") or os.environ.get("OKX_PROXY_URL")
             or _dotenv_value("BINANCE_PROXY_URL") or _dotenv_value("OKX_PROXY_URL"))
    if proxy:
        s.proxies.update({"http": proxy, "https": proxy})
    return s


def get(s: requests.Session, path: str, params: dict) -> list[dict] | list[list]:
    last = None
    for attempt in range(4):
        try:
            r = s.get(BASE + path, params=params, timeout=20)
            r.raise_for_status()
            payload = r.json()
            if isinstance(payload, dict) and payload.get("code"):
                raise RuntimeError(payload)
            return payload
        except (requests.RequestException, ValueError, RuntimeError) as exc:
            last = exc
            if attempt < 3:
                time.sleep(1.0 + attempt)
    raise RuntimeError(f"Binance {path} failed: {last}")


def update_klines(s: requests.Session, symbol: str, now_ms: int) -> str:
    rows = get(s, "/fapi/v1/klines", {"symbol": symbol, "interval": "15m", "limit": 1500})
    columns = ["ts", "open", "high", "low", "close", "volume", "close_time",
               "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore"]
    fresh = pd.DataFrame(rows, columns=columns)
    fresh["ts"] = pd.to_numeric(fresh["ts"], errors="coerce")
    fresh["close_time"] = pd.to_numeric(fresh["close_time"], errors="coerce")
    fresh = fresh.loc[fresh.close_time < now_ms, columns[:6] + columns[7:11]]
    for col in fresh.columns:
        fresh[col] = pd.to_numeric(fresh[col], errors="coerce")
    path = DATA / "klines" / f"{symbol}.csv.gz"
    old = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=fresh.columns)
    merged = pd.concat([old, fresh], ignore_index=True, sort=False).drop_duplicates("ts", keep="last").sort_values("ts")
    temporary = path.with_suffix(path.suffix + ".tmp")
    merged.to_csv(temporary, index=False, compression="gzip")
    temporary.replace(path)
    return pd.to_datetime(merged.ts.max(), unit="ms", utc=True).isoformat()


def update_metrics(s: requests.Session, symbol: str, now_ms: int) -> str:
    # Futures metrics endpoints cap this query at 500 observations (the
    # kline endpoint accepts 1500), which still covers more than 40 hours.
    params = {"symbol": symbol, "period": "5m", "limit": 500}
    endpoints = {
        "oi": ("/futures/data/openInterestHist", ("sum_open_interest", "sum_open_interest_value")),
        "top_account": ("/futures/data/topLongShortAccountRatio", ("count_toptrader_long_short_ratio",)),
        "top_position": ("/futures/data/topLongShortPositionRatio", ("sum_toptrader_long_short_ratio",)),
        "global": ("/futures/data/globalLongShortAccountRatio", ("count_long_short_ratio",)),
        "taker": ("/futures/data/takerlongshortRatio", ("sum_taker_long_short_vol_ratio",)),
    }
    frames = {}
    for name, (path, _) in endpoints.items():
        payload = get(s, path, params)
        frame = pd.DataFrame(payload)
        if frame.empty:
            continue
        frame["ts"] = pd.to_numeric(frame.get("timestamp"), errors="coerce")
        if name == "oi":
            frame["sum_open_interest"] = pd.to_numeric(frame.get("sumOpenInterest"), errors="coerce")
            frame["sum_open_interest_value"] = pd.to_numeric(frame.get("sumOpenInterestValue"), errors="coerce")
        elif name in {"top_account", "top_position", "global"}:
            target = endpoints[name][1][0]
            frame[target] = pd.to_numeric(frame.get("longShortRatio"), errors="coerce")
        else:
            frame["sum_taker_long_short_vol_ratio"] = pd.to_numeric(frame.get("buySellRatio"), errors="coerce")
        frames[name] = frame.set_index("ts")
    if not frames:
        raise RuntimeError(f"no Binance metrics returned for {symbol}")
    grid = pd.Index(sorted(set().union(*(set(x.index.dropna()) for x in frames.values()))), name="ts")
    raw = pd.DataFrame(index=grid)
    for frame in frames.values():
        raw = raw.join(frame[[c for c in frame.columns if c.startswith(("sum_", "count_"))]], how="outer")
    raw = raw.loc[raw.index < now_ms]
    raw.index = pd.to_datetime(raw.index, unit="ms", utc=True)
    grouped = raw.resample("15min", label="left", closed="left")
    out = grouped[[c for c in raw.columns if c != "sum_taker_long_short_vol_ratio"]].last()
    if "sum_taker_long_short_vol_ratio" in raw:
        out["sum_taker_long_short_vol_ratio"] = grouped["sum_taker_long_short_vol_ratio"].mean()
    out["n5m"] = grouped["sum_open_interest"].count() if "sum_open_interest" in raw else 1
    out = out.loc[out.n5m > 0].reset_index()
    # Pandas 3 preserves the millisecond resolution produced by resample;
    # dividing by 1e6 here would accidentally turn epoch milliseconds into
    # seconds and make the new rows sort before the existing archive.
    out["ts"] = out.ts.astype("int64")
    path = DATA / "metrics" / f"{symbol}.csv.gz"
    old = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=out.columns)
    merged = pd.concat([old, out], ignore_index=True, sort=False).drop_duplicates("ts", keep="last").sort_values("ts")
    temporary = path.with_suffix(path.suffix + ".tmp")
    merged.to_csv(temporary, index=False, compression="gzip")
    temporary.replace(path)
    return pd.to_datetime(merged.ts.max(), unit="ms", utc=True).isoformat()


def main() -> None:
    DATA.joinpath("klines").mkdir(parents=True, exist_ok=True)
    DATA.joinpath("metrics").mkdir(parents=True, exist_ok=True)
    s = session()
    now_ms = int(time.time() * 1000)
    status = {"updated_at": datetime.now(timezone.utc).isoformat(), "source": "binance_futures_rest", "symbols": {}}
    for symbol in SYMBOLS:
        status["symbols"][symbol] = {
            "klines_latest": update_klines(s, symbol, now_ms),
            "metrics_latest": update_metrics(s, symbol, now_ms),
        }
    temporary = STATUS.with_suffix(STATUS.suffix + ".tmp")
    temporary.write_text(json.dumps(status, ensure_ascii=True, indent=2), encoding="utf-8")
    temporary.replace(STATUS)
    print(json.dumps(status, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
