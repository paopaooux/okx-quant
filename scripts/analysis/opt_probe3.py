"""Re-pick the operating point under path accounting, at matched volatility.

The tail was chosen on `portfolio.py` numbers, which book a 10-hour trade's whole
P&L on one day.  That inflates daily volatility and manufactures single-day
drops, so it penalises the *frequent* cells hardest -- exactly the axis being
chosen.  Re-run the sweep on the mark-to-market path, and compare every cell
after scaling to the same 24% annual vol so the winner is not just the leveraged one.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from scripts.backtest.backtest_dir import simulate
from scripts.analysis.opt_probe import load_bars, bar_returns, occupancy, score, show, SYMS

RES = Path(__file__).resolve().parents[2] / "results" / "crypto"
TARGET_VOL = 0.242


def vol_matched(p: pd.Series, label: str, extra: dict | None = None) -> dict:
    d = (1 + p).resample("1D").prod() - 1
    k = TARGET_VOL / (d.std(ddof=1) * np.sqrt(365))
    out = score(p * k, label)
    out["lev"] = k
    if extra:
        out.update(extra)
    return out


if __name__ == "__main__":
    bars = load_bars()
    oos = pd.read_csv(RES / "oos_dir_c.csv.gz")
    oos["dt"] = pd.to_datetime(oos["dt"], utc=True)

    print("=== 分位阈值重扫（路径口径，统一 24% 波动）===")
    rows = []
    for tail in (0.10, 0.05, 0.02, 0.01, 0.005):
        tr = simulate(oos, "c", tail, "both", 10.0, "model", np.random.default_rng(11))
        r = bar_returns(tr, bars)
        o = occupancy(tr, bars)
        s0 = tr.dt.min()
        r = r.loc[r.index >= s0]; o = o.loc[o.index >= s0]
        p = (r * (1/3)).sum(axis=1)
        rows.append(vol_matched(p, f"tail {tail}", {"n": len(tr),
                                                    "exposure": o.sum(axis=1).mean()/3}))
    f = pd.DataFrame(rows).set_index("name")
    f["cagr"] = f.cagr.map("{:+.1%}".format); f["maxdd"] = f.maxdd.map("{:+.1%}".format)
    f["sharpe"] = f.sharpe.map("{:+.2f}".format); f["vol"] = f.vol.map("{:.1%}".format)
    f["exposure"] = f.exposure.map("{:.1%}".format); f["lev"] = f.lev.map("{:.2f}x".format)
    print(f.to_string())

    print("\n=== 三个币的贡献（tail 0.05，统一 24% 波动）===")
    tr = simulate(oos, "c", 0.05, "both", 10.0, "model", np.random.default_rng(11))
    s0 = tr.dt.min()
    r = bar_returns(tr, bars).loc[lambda d: d.index >= s0]
    rows = []
    for name, w in (("BTC+ETH+SOL 等权", (1/3, 1/3, 1/3)),
                    ("只 ETH+SOL", (0, .5, .5)),
                    ("BTC 半仓", (1/6, 5/12, 5/12)),
                    ("只 BTC", (1, 0, 0)),
                    ("只 ETH", (0, 1, 0)),
                    ("只 SOL", (0, 0, 1))):
        w = pd.Series(dict(zip(SYMS, w)))
        rows.append(vol_matched((r * w).sum(axis=1), name))
    f = pd.DataFrame(rows).set_index("name")
    f["cagr"] = f.cagr.map("{:+.1%}".format); f["maxdd"] = f.maxdd.map("{:+.1%}".format)
    f["sharpe"] = f.sharpe.map("{:+.2f}".format); f["vol"] = f.vol.map("{:.1%}".format)
    f["lev"] = f.lev.map("{:.2f}x".format)
    print(f.to_string())

    print("\n=== 随机对照（同笔数、同持仓，30 个种子，路径口径）===")
    sh, cg = [], []
    for seed in range(30):
        t2 = simulate(oos, "c", 0.05, "both", 10.0, "random", np.random.default_rng(seed))
        rr = bar_returns(t2, bars).loc[lambda d: d.index >= s0]
        s = score((rr * (1/3)).sum(axis=1), "rnd")
        sh.append(s["sharpe"]); cg.append(s["cagr"])
    print(f"  随机 夏普: 均值 {np.mean(sh):+.2f}  标准差 {np.std(sh):.2f}  "
          f"最大 {np.max(sh):+.2f}   模型 +1.58")
    print(f"  随机 年化: 均值 {np.mean(cg):+.1%}  最大 {np.max(cg):+.1%}   模型 +42.6%")
