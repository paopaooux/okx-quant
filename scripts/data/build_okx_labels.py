"""Re-label the panel on OKX prices, keeping the Binance-derived features.

The venue question is "can I trust a model fitted on Binance data if I execute
on OKX".  Splitting it: the features have no OKX equivalent (OKX retains 2 days
of 5m OI and 30 days at 1H, with no long/short or taker split), so they stay as
they are.  The *outcome* does have an OKX equivalent, and that is the half worth
testing -- if the same signal pays on OKX prices, the feed's venue does not
matter for anything but execution.

Measured divergence on BTC over 104,832 bars: median 0.71bps, std 3.3bps, 15m
return correlation 0.99911, against a ~100bps barrier.  So the expectation is
that labels barely move.  Running it anyway is the point -- an expectation is
not a result.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.data import build

DATA = Path(__file__).resolve().parents[2] / "data"


def main() -> None:
    panel = pd.read_csv(DATA / "panel.csv.gz")
    out = []
    for sym, g in panel.groupby("symbol", sort=True):
        g = g.sort_values("ts").reset_index(drop=True)
        okx = pd.read_csv(DATA / "okx_klines" / f"{sym}.csv.gz")
        okx = okx.drop_duplicates("ts").sort_values("ts")
        j = g[["ts"]].merge(okx[["ts", "open", "high", "low", "close"]], on="ts", how="left")
        miss = j["close"].isna().mean()

        o, h, l = (j[c].to_numpy(float) for c in ("open", "high", "low"))
        # Barrier width still comes from the Binance sigma already in the panel:
        # the width is a modelling choice made at decision time, not an outcome,
        # so changing its source would confound the venue test with a re-tuning.
        sigma = g["sigma"].to_numpy(float)
        flips = {}
        for name, kk, hh in build.CONFIGS:
            w = kk * sigma * np.sqrt(hh)
            lw, sw, held, ret = build.triple_barrier(o, h, l, w, hh)
            old_l = g[f"long_win_{name}"].to_numpy()
            old_s = g[f"short_win_{name}"].to_numpy()
            flips[name] = (float((lw != old_l).mean()), float((sw != old_s).mean()))
            g[f"long_win_{name}"] = lw
            g[f"short_win_{name}"] = sw
            g[f"held_{name}"] = held
            g[f"exit_ret_{name}"] = ret
        g["entry_px"] = pd.Series(o).shift(-1)
        g = g.dropna(subset=["entry_px"] + [f"long_win_{n}" for n, _, _ in build.CONFIGS])
        out.append(g)
        fl = "  ".join(f"{n}: {a:.2%}/{b:.2%}" for n, (a, b) in flips.items())
        print(f"{sym}: {len(g):,} rows, {miss:.3%} bars had no OKX quote   "
              f"label flips (long/short) {fl}", flush=True)

    p = pd.concat(out, ignore_index=True).sort_values(["ts", "symbol"]).reset_index(drop=True)
    p.to_csv(DATA / "panel_okx.csv.gz", index=False, compression="gzip")
    print(f"\npanel_okx: {len(p):,} rows -> {DATA/'panel_okx.csv.gz'}")


if __name__ == "__main__":
    main()
