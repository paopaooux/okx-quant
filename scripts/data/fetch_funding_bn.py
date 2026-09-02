"""Binance USDT-M funding-rate history, 3 years, + OKX-vs-Binance funding gap.

OKX's /public/funding-rate-history retains only ~3 months (286 settlements came
back for a 3-year request), so OKX-native funding for the sample does not exist --
the same retention wall as rubik OI.  Binance's /fapi/v1/fundingRate goes back to
listing, so the sample is built from Binance and the venue gap is *measured* on
the ~96 days where both exist, exactly as the price divergence was measured.
"""
import time, pathlib, urllib.request, json
import pandas as pd

OUT = pathlib.Path(__file__).resolve().parents[2] / "data" / "funding"; OUT.mkdir(parents=True, exist_ok=True)
START = int(pd.Timestamp("2023-09-01", tz="UTC").timestamp() * 1000)
UA = {"User-Agent": "Mozilla/5.0"}

def get(sym, start):
    u = (f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={sym}"
         f"&startTime={start}&limit=1000")
    with urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=30) as r:
        return json.load(r)

for sym in ("BTCUSDT", "ETHUSDT", "SOLUSDT"):
    rows, t = [], START
    while True:
        d = get(sym, t)
        if not d:
            break
        rows.extend(d)
        nt = int(d[-1]["fundingTime"]) + 1
        if nt <= t or len(d) < 1000:
            break
        t = nt
        time.sleep(0.15)
    f = pd.DataFrame(rows)
    f["ts"] = f["fundingTime"].astype("int64")
    f["rate"] = f["fundingRate"].astype(float)
    f = f[["ts", "rate"]].drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    f.to_csv(OUT / f"{sym}_bn.csv", index=False)
    print(f"{sym}: {len(f):,} settlements  "
          f"{pd.to_datetime(f.ts.iloc[0], unit='ms'):%Y-%m-%d} -> "
          f"{pd.to_datetime(f.ts.iloc[-1], unit='ms'):%Y-%m-%d}  "
          f"mean {f.rate.mean()*1e4:+.3f}bps/8h  pos {(f.rate>0).mean()*100:.1f}%")

print("\n--- OKX vs Binance funding, overlapping settlements ---")
for sym in ("BTCUSDT", "ETHUSDT", "SOLUSDT"):
    o = pd.read_csv(OUT / f"{sym}.csv"); b = pd.read_csv(OUT / f"{sym}_bn.csv")
    m = o.merge(b, on="ts", suffixes=("_okx", "_bn"))
    g = (m.rate_okx - m.rate_bn) * 1e4
    print(f"{sym}: n={len(m)}  OKX {m.rate_okx.mean()*1e4:+.3f} vs BN {m.rate_bn.mean()*1e4:+.3f} bps/8h  "
          f"gap mean {g.mean():+.3f} std {g.std():.3f}  corr {m.rate_okx.corr(m.rate_bn):.4f}")
