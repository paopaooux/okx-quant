"""OKX funding-rate history for the three swaps.

Why this exists: the backtest holds positions 3-11 hours, which spans one or two
8-hour funding settlements.  Funding is therefore a first-order cost (or credit)
at this horizon, and it is the one input that does NOT transfer from Binance --
the two venues' funding rates differ.  The strategy is short-biased, so if
funding was positive over the sample (longs paying shorts) the omission has been
working *against* the measured result, not for it.  Either way it must be
measured, not assumed.

Pagination note: /public/funding-rate-history walks backwards via `after`
(OKX's `after` means 'older than this ts'; `before` means 'newer than'),
100 records per page, ~3 settlements/day -> ~33 pages per instrument for 3 years.
"""
import sys, time, pathlib
import pandas as pd

sys.path.insert(0, "/home/haoyang.weng/myProgram")
from okx_quant.okx import OKXClient

INSTS = {"BTCUSDT": "BTC-USDT-SWAP", "ETHUSDT": "ETH-USDT-SWAP", "SOLUSDT": "SOL-USDT-SWAP"}
START_MS = int(pd.Timestamp("2023-09-01", tz="UTC").timestamp() * 1000)
OUT = pathlib.Path(__file__).resolve().parents[2] / "data" / "funding"; OUT.mkdir(parents=True, exist_ok=True)

c = OKXClient()
for sym, inst in INSTS.items():
    rows, before = [], None
    for _ in range(200):
        p = {"instId": inst, "limit": "100"}
        if before is not None:
            p["after"] = str(before)           # OKX: `after` = records OLDER than ts
        d = c._get("/api/v5/public/funding-rate-history", p)
        if isinstance(d, dict):
            d = d.get("data", [])
        if not d:
            break
        rows.extend(d)
        oldest = min(int(x["fundingTime"]) for x in d)
        if oldest <= START_MS:
            break
        before = oldest
        time.sleep(0.12)
    f = pd.DataFrame(rows)
    f["ts"] = f["fundingTime"].astype("int64")
    f["rate"] = f["realizedRate"].astype(float) if "realizedRate" in f else f["fundingRate"].astype(float)
    f = f[["ts", "rate"]].drop_duplicates("ts").sort_values("ts")
    f = f[f.ts >= START_MS].reset_index(drop=True)
    f.to_csv(OUT / f"{sym}.csv", index=False)
    print(f"{sym}: {len(f):,} settlements  "
          f"{pd.to_datetime(f.ts.iloc[0], unit='ms')} -> {pd.to_datetime(f.ts.iloc[-1], unit='ms')}  "
          f"mean {f.rate.mean()*1e4:+.3f}bps/8h  median {f.rate.median()*1e4:+.3f}  "
          f"pos {(f.rate>0).mean()*100:.1f}%")
