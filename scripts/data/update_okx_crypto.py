"""Persist an OKX-native crypto tail for portability checks and monitoring.

This archive is intentionally separate from Binance training data.  Candles
are retained from ``OKX_CRYPTO_START`` onward; Rubik 5m metrics are appended as
far back as OKX currently exposes them (roughly two days).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from strategies.stocks.market.okx import OKXClient, OKXError

DATA = Path(os.environ.get("OKX_CRYPTO_DATA_DIR", ROOT / "data"))
KLINE_DIR = DATA / "okx_klines"
METRIC_DIR = DATA / "okx_metrics"
STATUS = Path(os.environ.get("OKX_CRYPTO_STATUS", DATA / "okx_crypto_status.json"))
SYMBOLS = {
    "ADAUSDT": "ADA-USDT-SWAP", "BNBUSDT": "BNB-USDT-SWAP",
    "BTCUSDT": "BTC-USDT-SWAP", "DOGEUSDT": "DOGE-USDT-SWAP",
    "ETHUSDT": "ETH-USDT-SWAP", "LINKUSDT": "LINK-USDT-SWAP",
    "SOLUSDT": "SOL-USDT-SWAP", "XRPUSDT": "XRP-USDT-SWAP",
}
# Keep enough OKX contract history for the longest rolling feature (384 x 15m
# bars is four days). An explicit start remains available for reproducible
# backfills; live refreshes default to a moving 30-day window.
HISTORY_DAYS = max(7, int(os.environ.get("OKX_CRYPTO_HISTORY_DAYS", "30")))
_start_raw = os.environ.get("OKX_CRYPTO_START")
START = (pd.to_datetime(_start_raw, utc=True)
         if _start_raw else pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=HISTORY_DAYS))
INTERVAL = max(60, int(os.environ.get("OKX_CRYPTO_INTERVAL", "300")))


def _write(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False, compression="gzip")
    tmp.replace(path)


def update_klines(client: OKXClient, symbol: str, inst: str) -> str:
    path = KLINE_DIR / f"{symbol}.csv.gz"
    old = pd.read_csv(path) if path.exists() else pd.DataFrame()
    start_ms = int(START.timestamp() * 1000)
    if not old.empty:
        old_end = int(pd.to_datetime(old.ts, utc=True).max().timestamp() * 1000)
        start_ms = max(start_ms, old_end - 2 * 86400 * 1000)
    fresh = client.history_candles(inst, "15m", start_ms, max_pages=20)
    # Pandas 3 keeps the datetime at millisecond resolution; dividing by 1e6
    # would turn epoch milliseconds into seconds (1970 timestamps).
    fresh["ts"] = pd.to_datetime(fresh["ts"], utc=True).astype("int64")
    merged = pd.concat([old, fresh], ignore_index=True, sort=False) if not old.empty else fresh
    merged = merged.drop_duplicates("ts", keep="last").sort_values("ts").reset_index(drop=True)
    _write(path, merged)
    return pd.to_datetime(merged.ts.max(), unit="ms", utc=True).isoformat()


def update_metrics(client: OKXClient, symbol: str, inst: str) -> str:
    path = METRIC_DIR / f"{symbol}.csv.gz"
    rows = []
    for endpoint, params in (
        ("/api/v5/rubik/stat/contracts/open-interest-history", {"instId": inst, "period": "5m", "limit": "500"}),
        ("/api/v5/rubik/stat/contracts/long-short-account-ratio", {"ccy": symbol[:-4], "instType": "SWAP", "period": "5m", "limit": "500"}),
    ):
        payload = client._get(endpoint, params)
        source = endpoint.rsplit("/", 1)[-1]
        for row in payload:
            # Rubik endpoints return compact arrays, not JSON objects.
            if source == "open-interest-history":
                values = list(row) + [None] * 4
                item = {"ts": values[0], "sum_open_interest": values[1],
                        "oi_ccy": values[2], "sum_open_interest_value": values[3]}
            else:
                values = list(row) + [None] * 2
                item = {"ts": values[0], "count_long_short_ratio": values[1]}
            item["source"] = source
            rows.append(item)
    fresh = pd.DataFrame(rows)
    if fresh.empty:
        raise OKXError(f"no OKX Rubik metrics for {symbol}")
    fresh["ts"] = pd.to_numeric(fresh.get("ts"), errors="coerce")
    fresh["inst_id"] = inst
    old = pd.read_csv(path) if path.exists() else pd.DataFrame()
    merged = pd.concat([old, fresh], ignore_index=True, sort=False) if not old.empty else fresh
    merged = merged.drop_duplicates(["source", "ts"], keep="last").sort_values("ts").reset_index(drop=True)
    _write(path, merged)
    return pd.to_datetime(merged.ts.max(), unit="ms", utc=True).isoformat()


def run_once(status: dict | None = None) -> dict:
    client = OKXClient(timeout=20)
    status = status or {"source": "okx", "start": START.isoformat(), "symbols": {}}
    status["updated_at"] = datetime.now(timezone.utc).isoformat()
    for symbol, inst in SYMBOLS.items():
        result = status["symbols"].setdefault(symbol, {})
        try:
            result["klines_latest"] = update_klines(client, symbol, inst)
            result["klines_error"] = None
        except Exception as exc:
            result["klines_error"] = str(exc)
        try:
            result["metrics_latest"] = update_metrics(client, symbol, inst)
            result["metrics_error"] = None
        except Exception as exc:
            result["metrics_error"] = str(exc)
    _write_status(status)
    print(json.dumps(status, ensure_ascii=True), flush=True)
    return status


def _write_status(status: dict) -> None:
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATUS.with_suffix(STATUS.suffix + ".tmp")
    tmp.write_text(json.dumps(status, ensure_ascii=True, indent=2), encoding="utf-8")
    tmp.replace(STATUS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    status = None
    while True:
        started = time.monotonic()
        status = run_once(status)
        if args.once:
            return
        time.sleep(max(1.0, INTERVAL - (time.monotonic() - started)))


if __name__ == "__main__":
    main()
