"""Public-data OKX research in an isolated archive. Never imports live trading."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from strategies.stocks.market.okx import OKXClient
from scripts.combinations.run import QUANT_SYMBOLS

BAR_MS = 900000
SOURCES = {
    "oi": ("/api/v5/rubik/stat/contracts/open-interest-history",
           ["ts", "sum_open_interest", "oi_ccy", "sum_open_interest_value"]),
    "top_account": ("/api/v5/rubik/stat/contracts/long-short-account-ratio-contract-top-trader",
                    ["ts", "count_toptrader_long_short_ratio"]),
    "top_position": ("/api/v5/rubik/stat/contracts/long-short-position-ratio-contract-top-trader",
                     ["ts", "sum_toptrader_long_short_ratio"]),
    "global": ("/api/v5/rubik/stat/contracts/long-short-account-ratio-contract",
               ["ts", "count_long_short_ratio"]),
    "taker": ("/api/v5/rubik/stat/taker-volume-contract", ["ts", "sell_quote_volume", "buy_quote_volume"]),
}


def epoch_ms(stamp):
    return int(pd.Timestamp(stamp).timestamp() * 1000)


def save_csv(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, compression="gzip")
    temporary.replace(path)


def coverage(frame, start, end):
    ts = pd.to_numeric(frame.ts).astype("int64")
    grid = np.arange(epoch_ms(start), epoch_ms(end), BAR_MS)
    missing = np.setdiff1d(grid, ts.to_numpy())
    return dict(rows=len(frame), start=pd.to_datetime(ts.min(), unit="ms", utc=True).isoformat(),
                end=pd.to_datetime(ts.max(), unit="ms", utc=True).isoformat(),
                requested_bars=len(grid), missing_bars=len(missing))


def download_candles(client, symbol, folder, start, end):
    path = folder / "klines" / f"{symbol}.csv.gz"
    seed = path if path.exists() else Path("data/okx_klines") / f"{symbol}.csv.gz"
    frame = pd.read_csv(seed) if seed.exists() else pd.DataFrame()
    inst = symbol.replace("USDT", "-USDT-SWAP")
    floor, ceiling = epoch_ms(start), epoch_ms(end)
    if not frame.empty:
        frame = frame.loc[frame.ts.ge(floor) & frame.ts.lt(ceiling)]
    # Fetch the current tail once, then page backwards from the oldest known
    # row. The live archive is read only; partial research downloads resume.
    cursors = [ceiling]
    if len(frame):
        cursors.append(int(frame.ts.min()))
    for cursor in cursors:
        for page in range(1000):
            rows = client._get("/api/v5/market/history-candles",
                               {"instId": inst, "bar": "15m", "after": str(cursor), "limit": "300"})
            if not rows:
                break
            raw = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume",
                                              "volume_ccy", "volume_quote", "confirm"])
            raw = raw.loc[raw.confirm.astype(str).eq("1")].drop(columns="confirm")
            raw = raw.apply(pd.to_numeric, errors="coerce")
            oldest = min(int(r[0]) for r in rows)
            raw = raw.loc[raw.ts.ge(floor) & raw.ts.lt(ceiling)]
            frame = pd.concat([frame, raw], ignore_index=True).drop_duplicates("ts", keep="last").sort_values("ts")
            if page % 10 == 0:
                save_csv(frame, path)
                print(f"candles {symbol}: {len(frame)} rows, earliest={pd.to_datetime(frame.ts.min(), unit='ms', utc=True)}", flush=True)
            if oldest <= floor or oldest >= cursor:
                break
            # The first pass only appends the tail when a seed already covers it.
            if len(cursors) == 2 and cursor == cursors[0]:
                break
            cursor = oldest
            time.sleep(.15)
    save_csv(frame, path)
    info = coverage(frame, start, end)
    print(f"candles complete {symbol}: {info}", flush=True)
    if info["missing_bars"]:
        raise ValueError(f"Incomplete OKX candles for {symbol}: {info}")
    return info


def download_metric(client, symbol, source, folder, start, end):
    endpoint, columns = SOURCES[source]
    path = folder / "metrics" / f"{symbol}_{source}_15m.csv.gz"
    done_path = path.with_suffix(".coverage.json")
    if path.exists() and done_path.exists():
        return json.loads(done_path.read_text())
    cursor = epoch_ms(end) - 1
    batches = []
    for page in range(30):
        params = dict(instId=symbol.replace("USDT", "-USDT-SWAP"), period="15m", limit="100", end=str(cursor))
        if source == "taker":
            params["unit"] = "2"
        rows = client._get(endpoint, params)
        if not rows:
            break
        raw = pd.DataFrame(rows, columns=columns).apply(pd.to_numeric, errors="coerce")
        batches.append(raw)
        oldest = int(raw.ts.min())
        if oldest <= epoch_ms(start) or oldest >= cursor:
            break
        cursor = oldest - 1
        time.sleep(.45)
    if not batches:
        raise ValueError(f"No recent {source} data for {symbol}")
    frame = pd.concat(batches, ignore_index=True).drop_duplicates("ts").sort_values("ts")
    frame = frame.loc[frame.ts.ge(epoch_ms(start)) & frame.ts.lt(epoch_ms(end))]
    save_csv(frame, path)
    info = coverage(frame, start, end)
    info.update(source=source, endpoint=endpoint, period="15m", symbol=symbol,
                note="Unavailable older observations remain missing, never interpolated from 4H/1D.")
    done_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"metrics {symbol}/{source}: {info['rows']} rows from {info['start']}", flush=True)
    return info


def probe_history(client, folder):
    results = []
    for source, (endpoint, _) in SOURCES.items():
        for period in ("15m", "4H"):
            params = dict(instId="BTC-USDT-SWAP", period=period, limit="100",
                          end=str(epoch_ms("2026-03-04T00:00:00Z")))
            if source == "taker":
                params["unit"] = "2"
            rows = client._get(endpoint, params)
            results.append(dict(source=source, period=period, params=params, endpoint=endpoint, rows=rows))
            time.sleep(.45)
    (folder / "history_probes.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return [{k: v for k, v in r.items() if k != "rows"} | {"row_count": len(r["rows"])} for r in results]


def download(folder):
    folder.mkdir(parents=True, exist_ok=True)
    manifest_path = folder / "download_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        manifest = dict(start="2026-02-01T00:00:00Z", end=pd.Timestamp.now(tz="UTC").floor("15min").isoformat(),
                        source="OKX public APIs", symbols=list(QUANT_SYMBOLS), candles={}, metrics={},
                        live_modified=False)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    client = OKXClient(timeout=20)
    start, end = pd.Timestamp(manifest["start"]), pd.Timestamp(manifest["end"])
    for symbol in QUANT_SYMBOLS:
        manifest["candles"][symbol] = download_candles(client, symbol, folder, start, end)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for symbol in QUANT_SYMBOLS:
        for source in SOURCES:
            manifest["metrics"][f"{symbol}/{source}"] = download_metric(client, symbol, source, folder, start, end)
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["history_probes"] = probe_history(client, folder)
    manifest["complete"] = True
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["download"])
    parser.add_argument("--data", type=Path, default=Path("data/research_okx_native_20260907"))
    args = parser.parse_args()
    download(args.data)


if __name__ == "__main__":
    main()
