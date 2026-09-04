"""Build a deduplicated OKX stock-perpetual archive for the stock strategy.

Live execution uses ``AAPL-USDT-SWAP`` perpetuals, which have independent
prices, contract sizing and short-side liquidity.  This updater owns that
contract universe and its candles.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from strategies.stocks.market.okx import OKXClient, OKXError, update_cache
from strategies.stocks.market.universe_tech import TECH

TARGET = Path(os.environ.get("STOCK_SWAP_DATA_DIR", ROOT / "data" / "stocks_swap"))
HISTORY_DAYS = max(7, int(os.environ.get("STOCK_SWAP_HISTORY_DAYS", "400")))
MAX_PAGES = max(20, int(os.environ.get("STOCK_SWAP_MAX_PAGES", "500")))
WORKERS = max(1, int(os.environ.get("STOCK_SWAP_WORKERS", "4")))


def discover(client: OKXClient) -> pd.DataFrame:
    rows = client._get("/api/v5/public/instruments", {"instType": "SWAP"})
    records = []
    for row in rows:
        if (row.get("instCategory") != "3" or row.get("state") != "live"
                or row.get("settleCcy") != "USDT" or row.get("ctType") != "linear"):
            continue
        inst = str(row.get("instId") or "")
        ticker = str(row.get("ctValCcy") or "")
        if not inst.endswith("-USDT-SWAP") or not ticker:
            continue
        records.append({
            "instId": inst,
            "ticker": ticker,
            "baseCcy": ticker,
            "quoteCcy": "USDT",
            "state": row.get("state"),
            "instCategory": row.get("instCategory"),
            "lever": row.get("lever"),
            "ctVal": row.get("ctVal"),
            "ctValCcy": row.get("ctValCcy"),
            "lotSz": row.get("lotSz"),
            "minSz": row.get("minSz"),
            "tickSz": row.get("tickSz"),
            "list_ts": pd.to_datetime(
                pd.to_numeric(row.get("listTime"), errors="coerce"),
                unit="ms", utc=True, errors="coerce"
            ),
            "kind": "single",
        })
    frame = pd.DataFrame(records)
    if frame.empty:
        raise RuntimeError("OKX returned no live stock perpetuals")
    # One contract per underlying ticker.  The exchange's instId is the
    # canonical key; sorting makes the choice deterministic if aliases appear.
    frame = (frame.sort_values(["ticker", "instId"])
             .drop_duplicates("ticker", keep="first")
             .reset_index(drop=True))
    return frame


def write_universe(frame: pd.DataFrame) -> None:
    TARGET.mkdir(parents=True, exist_ok=True)
    frame.to_csv(TARGET / "universe.csv", index=False)


def refresh_candles(client: OKXClient, frame: pd.DataFrame, symbols: set[str] | None = None) -> dict:
    done, failed = [], []
    selected = frame if symbols is None else frame.loc[frame.ticker.isin(symbols)]

    def refresh_one(row: object) -> tuple[str, str, str | None]:
        # requests.Session is not shared between workers: each client owns its
        # connection pool and retry state, while each ticker has its own file.
        try:
            worker_client = OKXClient(timeout=30, proxy_url=client.proxy_url)
            update_cache(worker_client, row.instId, "5m", HISTORY_DAYS, TARGET, max_pages=MAX_PAGES)
            return row.ticker, row.instId, None
        except (OKXError, OSError, ValueError) as exc:
            return row.ticker, row.instId, str(exc)

    with ThreadPoolExecutor(max_workers=min(WORKERS, len(selected) or 1)) as pool:
        futures = [pool.submit(refresh_one, row) for row in selected.itertuples(index=False)]
        for future in as_completed(futures):
            ticker, inst_id, error = future.result()
            if error is None:
                done.append(ticker)
                print(f"[stocks-swap] {ticker} {inst_id} refreshed", flush=True)
            else:
                failed.append({"ticker": ticker, "instId": inst_id, "error": error})
                print(f"[stocks-swap] {ticker} failed: {error}", flush=True)
    status = {
        "source": "OKX_SWAP",
        "history_days": HISTORY_DAYS,
        "workers": WORKERS,
        "symbols": len(selected),
        "refreshed": done,
        "failed": failed,
        "updated_at": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    (TARGET / "data_status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discover-only", action="store_true")
    parser.add_argument("--tickers", nargs="*", help="只刷新这些底层 ticker；默认刷新全部")
    parser.add_argument("--tech-only", action="store_true", help="只刷新当前 TECH 策略池")
    args = parser.parse_args()
    client = OKXClient(timeout=30)
    universe = discover(client)
    write_universe(universe)
    print(f"[stocks-swap] universe={len(universe)} target={TARGET}", flush=True)
    if args.discover_only:
        return 0
    symbols = set(args.tickers) if args.tickers else None
    if args.tech_only:
        symbols = (symbols & TECH) if symbols is not None else set(TECH)
    refresh_candles(client, universe, symbols)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
