"""Out-of-sample performance split by market regime.

The 3-year sample's OOS window (2024-11 -> 2026-08) was net-down in all three
symbols, so a short-only signal could not be falsified there: quarterly analysis
showed 79% of the total return came from two sharp down quarters and the one
strong up quarter (2025Q3, ETH +66%) lost -45bps/trade.  This script re-asks the
same question on the extended sample, whose OOS window now contains the 2023
recovery and the 2024 bull.

Every quarter is reported next to a `random` control of the SAME trade count and
side, so a quarter that merely rode beta is separable from one where selection
added something.

Usage: python3 -m scripts.backtest.regime [tag] [cfg] [tail] [policy]
"""
from __future__ import annotations

import sys
import numpy as np
import pandas as pd
from pathlib import Path

from scripts.backtest import backtest_dir as bd
from scripts.backtest.backtest import summarise

ROOT = Path(__file__).resolve().parents[2]
tag = sys.argv[1] if len(sys.argv) > 1 else "b"
cfg = sys.argv[2] if len(sys.argv) > 2 else tag.split("_")[0]
tail = float(sys.argv[3]) if len(sys.argv) > 3 else 0.01
pol = sys.argv[4] if len(sys.argv) > 4 else "short"
COST = 10.0

oos = pd.read_csv(ROOT / "results" / "crypto" / f"oos_dir_{tag}.csv.gz")
oos["dt"] = pd.to_datetime(oos["dt"], utc=True)
years = (oos.dt.max() - oos.dt.min()).total_seconds() / (365.25 * 86400)

tr = bd.simulate(oos, cfg, tail, pol, COST, "model", np.random.default_rng(11))
tr["q"] = tr["dt"].dt.to_period("Q")

# Control: 30 random draws, averaged, so a single lucky draw cannot masquerade
# as the benchmark (the 54-cell null showed one seed reaching the 97th pct).
ctl = []
for s in range(30):
    c = bd.simulate(oos, cfg, tail, pol, COST, "random", np.random.default_rng(2000 + s))
    c["q"] = c["dt"].dt.to_period("Q")
    ctl.append(c)

px = pd.read_csv(ROOT / "data" / "panel.csv.gz", usecols=["ts", "dt", "symbol", "entry_px"])
px["dt"] = pd.to_datetime(px["dt"], utc=True)
px["q"] = px["dt"].dt.to_period("Q")

print(f"{tag}  cfg={cfg}  tail={tail}  {pol}  cost={COST}bps   "
      f"OOS {oos.dt.min():%Y-%m} -> {oos.dt.max():%Y-%m} ({years:.2f}y)\n")
hdr = (f"{'quarter':>8} {'BTC%':>7} {'ETH%':>7} {'SOL%':>7} | {'n':>5} {'win%':>6} "
       f"{'net':>8} {'t':>6} | {'rand net':>9}")
print(hdr); print("-" * len(hdr))
for q, g in tr.groupby("q"):
    s = summarise(g, 1.0)
    mv = {}
    for sym, h in px[px.q == q].groupby("symbol"):
        h = h.sort_values("ts")
        mv[sym] = h["entry_px"].iloc[-1] / h["entry_px"].iloc[0] - 1
    rn = np.mean([summarise(c[c.q == q], 1.0)["net_bps"] for c in ctl
                  if len(c[c.q == q])])
    print(f"{str(q):>8} {mv.get('BTCUSDT',np.nan):>+6.1%} {mv.get('ETHUSDT',np.nan):>+6.1%} "
          f"{mv.get('SOLUSDT',np.nan):>+6.1%} | {s['trades']:>5.0f} {s['winrate']:>5.1%} "
          f"{s['net_bps']:>+8.1f} {s['t_stat']:>+6.2f} | {rn:>+9.1f}")

# Up-quarter vs down-quarter aggregate, defined on the equal-weight basket so the
# split does not depend on which symbol a trade happened to be in.
basket = {}
for q, g in px.groupby("q"):
    r = [h.sort_values("ts")["entry_px"].iloc[-1] / h.sort_values("ts")["entry_px"].iloc[0] - 1
         for _, h in g.groupby("symbol")]
    basket[q] = float(np.mean(r))
tr["up"] = tr["q"].map(basket) > 0
print()
for up, g in tr.groupby("up"):
    s = summarise(g, 1.0)
    rn = np.mean([summarise(c[c["q"].map(basket).gt(0) == up], 1.0)["net_bps"] for c in ctl])
    print(f"{'涨季' if up else '跌季':>6}  n={s['trades']:>4}  win {s['winrate']:>5.1%}  "
          f"net {s['net_bps']:>+7.1f}bps  t {s['t_stat']:>+5.2f}   随机对照 {rn:>+6.1f}bps")

tr = tr.sort_values("dt")
k = max(1, len(tr) // 100)
top = tr.nlargest(10, "net")["net"].sum()
print(f"\n集中度: 前10笔占总收益 {top / tr['net'].sum():.1%}   "
      f"中位数 {tr['net'].median()*1e4:+.1f}bps   均值 {tr['net'].mean()*1e4:+.1f}bps")
