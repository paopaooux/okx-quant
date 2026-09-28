"""持续刷新 OKX 股票永续的 5 分钟 K 线。

股票信号只使用本地 OKX 行情生成休市偏离事件；本进程不抓取外部公告，
也没有交易凭证和下单权限。
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from strategies.stocks.market.okx import OKXClient, OKXError, update_cache
from scripts.live.combination_policy import stock_instruments

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("AUTO_STOCK_DATA", ROOT / "data" / "stocks_swap"))
UNIVERSE = Path(os.environ.get("AUTO_STOCK_UNIVERSE", DATA / "universe.csv"))
INTERVAL = max(60, int(os.environ.get("STOCK_CANDLE_INTERVAL", "60")))
HISTORY_DAYS = max(18, int(os.environ.get("STOCK_HISTORY_DAYS", "18")))
STATUS = Path(os.environ.get("AUTO_STOCK_STATUS", DATA / "data_status.json"))


def refresh_candles(client: OKXClient, on_progress=None) -> int:
    if not UNIVERSE.exists():
        return 0
    universe = pd.read_csv(UNIVERSE)
    count = failures = 0
    eligible = stock_instruments(universe)
    for inst in eligible:
        try:
            update_cache(client, inst, "5m", HISTORY_DAYS, DATA)
            count += 1
            if on_progress and (count == 1 or count % 10 == 0):
                on_progress(count)
        except (OKXError, OSError, ValueError) as exc:
            failures += 1
            print(f"candle {inst} refresh failed: {exc}", flush=True)
    if len(eligible) and failures == len(eligible):
        raise RuntimeError(f"all {len(eligible)} stock candle requests failed")
    return count


def _write_status(status: dict) -> None:
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATUS.with_suffix(STATUS.suffix + ".tmp")
    temporary.write_text(json.dumps(status, ensure_ascii=True, indent=2), encoding="utf-8")
    temporary.replace(STATUS)


def _latest_candle_ts() -> str | None:
    newest = None
    paths = (DATA / "5m").glob("*.csv") if (DATA / "5m").exists() else []
    for path in paths:
        try:
            tail = pd.read_csv(path, usecols=["ts"]).tail(1)
            if tail.empty:
                continue
            stamp = pd.to_datetime(tail.iloc[0]["ts"], utc=True, errors="coerce")
            if pd.notna(stamp) and (newest is None or stamp > newest):
                newest = stamp
        except (OSError, ValueError, KeyError):
            continue
    return newest.isoformat() if newest is not None else None


def run_once(do_candles: bool = True, status: dict | None = None) -> dict:
    status = status or {}
    stamp = datetime.now(timezone.utc).isoformat()
    status.update({"service": "okx-stock-data", "updated_at": stamp,
                   "candle_interval_s": INTERVAL})
    if do_candles:
        client = OKXClient(timeout=20)
        def publish(count):
            completed = datetime.now(timezone.utc).isoformat()
            status["updated_at"] = completed
            status["candles"] = {"last_success_at": completed, "last_error": None,
                                 "refreshed": count, "refreshing": True}
            _write_status(status)
        try:
            count = refresh_candles(client, on_progress=publish)
            completed = datetime.now(timezone.utc).isoformat()
            status["updated_at"] = completed
            status["candles"] = {"last_success_at": completed, "last_error": None,
                                  "refreshed": count, "refreshing": False,
                                  "latest_bar_at": _latest_candle_ts()}
        except Exception as exc:  # keep the daemon alive across transient outages
            status.setdefault("candles", {}).update({"last_error": str(exc), "last_failed_at": stamp})
            print(f"candle refresh failed: {exc}", flush=True)
        finally:
            client.session.close()
            status.setdefault("candles", {})["refreshing"] = False
        _write_status(status)
    print(json.dumps({"ts": stamp,
                      "candles_refreshed": status.get("candles", {}).get("refreshed", 0)
                      if do_candles else None,
                      "candles_due": do_candles, "candle_interval_s": INTERVAL},
                     ensure_ascii=True), flush=True)
    return status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="refresh once and exit")
    args = parser.parse_args()
    status: dict = {}
    while True:
        status = run_once(status=status)
        if args.once:
            return
        # Align polls just after clock boundaries instead of preserving an
        # arbitrary startup offset relative to five-minute candle closes.
        time.sleep(next_refresh_delay(time.time()))


def next_refresh_delay(now):
    return max(1.0, ((now - 2) // INTERVAL + 1) * INTERVAL + 2 - now)


if __name__ == "__main__":
    main()
