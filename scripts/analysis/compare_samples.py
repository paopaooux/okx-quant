"""Did the longer history help, or did the OOS window just get luckier?

The extended run reports better numbers than the 3-year run, but its OOS window
is different, so the two headline numbers are not comparable.  This isolates the
one thing that IS comparable: the calendar window both studies scored, 2024-11 ->
2026-08.  Same bars, same cells, same cost -- the only difference is how much
history the model was fitted on.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.backtest import backtest_dir as bd
from scripts.backtest.backtest import summarise

SPLIT = pd.Timestamp("2024-11-01", tz="UTC")
CELLS = [("c", 0.05, "both"), ("c", 0.02, "both"), ("c", 0.01, "short"),
         ("b", 0.02, "both"), ("b", 0.01, "short"), ("a", 0.02, "short")]

hdr = (f"{'cell':>16} | {'3年模型 n':>10} {'win%':>6} {'net':>7} {'t':>6} "
       f"| {'5.7年模型 n':>11} {'win%':>6} {'net':>7} {'t':>6}")
print("共同评分窗口 2024-11 -> 2026-08（两次研究都覆盖），唯一差别是训练历史长度\n")
print(hdr); print("-" * len(hdr))
for cfg, tl, pol in CELLS:
    line = []
    for root in ("archive_3y/results", "results"):
        o = pd.read_csv(f"{root}/oos_dir_{cfg}.csv.gz")
        o["dt"] = pd.to_datetime(o["dt"], utc=True)
        tr = bd.simulate(o, cfg, tl, pol, 10.0, "model", np.random.default_rng(11))
        tr = tr[tr.dt >= SPLIT]
        line.append(summarise(tr, 1.0) if len(tr) else None)
    if None in line:
        continue
    a, b = line
    print(f"{cfg+' t='+str(tl)+' '+pol:>16} | {a['trades']:>10.0f} {a['winrate']:>5.1%} "
          f"{a['net_bps']:>+7.1f} {a['t_stat']:>+6.2f} | {b['trades']:>11.0f} "
          f"{b['winrate']:>5.1%} {b['net_bps']:>+7.1f} {b['t_stat']:>+6.2f}")
