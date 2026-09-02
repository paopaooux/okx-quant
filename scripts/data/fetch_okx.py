"""Pull OKX USDT-swap 15m candles for the three majors.

Purpose is a venue-portability test, not a second data source.  The features
stay Binance-derived because OKX has no archive for them -- its own rubik OI
endpoint retains 2 days at 5m and 30 days at 1H, and exposes only aggregate
oi_usd/vol_usd per currency with no long/short or taker split.  Three years of
OKX-native positioning data does not exist and cannot be reconstructed.

What OKX *can* answer is whether the outcome changes when the trade is priced
on the venue it would actually be executed on.  So: Binance metrics in, OKX
prices out.  If the edge survives that swap, the Binance feed is a legitimate
input regardless of where the order goes.

Only the OKXClient HTTP plumbing is reused (TLS1.2 adapter, browser UA, rate
limiting) -- no strategy code from okx_quant is imported.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, "/home/haoyang.weng/myProgram")
from okx_quant.okx import OKXClient          # noqa: E402  (plumbing only)

OUT = Path(__file__).resolve().parents[2] / "data" / "okx_klines"
INSTS = {"BTCUSDT": "BTC-USDT-SWAP", "ETHUSDT": "ETH-USDT-SWAP", "SOLUSDT": "SOL-USDT-SWAP"}
START = "2023-09-01"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    c = OKXClient()
    start_ms = int(pd.Timestamp(START, tz="UTC").timestamp() * 1000)
    for sym, inst in INSTS.items():
        path = OUT / f"{sym}.csv.gz"
        if path.exists():
            print(f"{sym}: exists, skip", flush=True)
            continue
        t0 = time.time()
        df = c.history_candles(inst, "15m", start_ms, max_pages=500)
        df = df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
        df["ts"] = (pd.to_datetime(df["ts"], utc=True).astype("int64") // 10**6)
        df.to_csv(path, index=False, compression="gzip")
        span = pd.to_datetime(df["ts"], unit="ms", utc=True)
        print(f"{sym} <- {inst}: {len(df):,} bars  {span.min():%Y-%m-%d} -> {span.max():%Y-%m-%d}"
              f"  [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
