"""Sizing levers that can move risk-adjusted return, not just scale it.

opt_probe showed pooling idle capital only adds leverage: Sharpe falls as the
notional rises.  The levers that can actually move Sharpe are the ones that make
the *risk* of each trade comparable -- equal dollars on BTC and SOL is not equal
risk when SOL's barrier is 70% wider -- and the ones that put more capital
behind the trades the model is more sure about.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from scripts.backtest.backtest_dir import simulate
from scripts.analysis.opt_probe import load_bars, bar_returns, occupancy, score, show, SYMS

RES = Path(__file__).resolve().parents[2] / "results" / "crypto"


def trades_with_meta(cfg="c", tail=0.05, cost=10.0):
    oos = pd.read_csv(RES / f"oos_dir_{cfg}.csv.gz")
    oos["dt"] = pd.to_datetime(oos["dt"], utc=True)
    tr = simulate(oos, cfg, tail, "both", cost, "model", np.random.default_rng(11))
    meta = oos.set_index(["symbol", "ts"])[[f"width_{cfg}", "p_up",
                                            f"hi_{tail}", f"lo_{tail}"]]
    tr = tr.join(meta, on=["symbol", "ts"])
    # Distance past the fold's own threshold, in units of the gap to 0.5.
    hi, lo, p = tr[f"hi_{tail}"], tr[f"lo_{tail}"], tr["p_up"]
    tr["edge"] = np.where(tr.side > 0, (p - hi) / (hi - 0.5), (lo - p) / (0.5 - lo))
    tr["width"] = tr[f"width_{cfg}"]
    return tr


def weighted_bars(tr, bars, wcol):
    """Per-bar returns with a per-trade notional multiplier."""
    parts = []
    for mult, g in tr.groupby(wcol):
        parts.append(bar_returns(g, bars) * mult)
    out = sum(parts)
    return out


if __name__ == "__main__":
    bars = load_bars()
    tr = trades_with_meta()
    start = tr.dt.min()

    r0 = bar_returns(tr, bars); r0 = r0.loc[r0.index >= start]
    occ = occupancy(tr, bars).loc[lambda d: d.index >= start]
    rows = [score((r0 * (1/3)).sum(axis=1), "基线 等名义 1/3", occ.sum(axis=1)/3)]

    med = tr["width"].median()
    print(f"屏障宽度中位数 (bps): " + "  ".join(
        f"{s}={tr[tr.symbol==s]['width'].median()*1e4:.0f}" for s in SYMS))
    print(f"信号强度 edge: p10={tr.edge.quantile(.1):.2f} "
          f"p50={tr.edge.median():.2f} p90={tr.edge.quantile(.9):.2f}")

    # A. equal risk: notional inversely proportional to the trade's own barrier
    #    width, clipped so no single trade can run away.
    tr["w_risk"] = (med / tr["width"]).clip(0.5, 2.0).round(2)
    rA = weighted_bars(tr, bars, "w_risk").loc[lambda d: d.index >= start]
    oA = occupancy(tr, bars).mul(0).add(0)  # exposure recomputed below
    gA = sum(occupancy(g, bars) * m for m, g in tr.groupby("w_risk")).loc[lambda d: d.index >= start]
    rows.append(score((rA * (1/3)).sum(axis=1), "A 等风险(宽度反比)", gA.sum(axis=1)/3))

    # B. conviction sizing: more notional the further past the fold's threshold.
    tr["w_edge"] = (0.5 + tr.edge.clip(0, 3) / 1.5).clip(0.5, 2.0).round(2)
    rB = weighted_bars(tr, bars, "w_edge").loc[lambda d: d.index >= start]
    gB = sum(occupancy(g, bars) * m for m, g in tr.groupby("w_edge")).loc[lambda d: d.index >= start]
    rows.append(score((rB * (1/3)).sum(axis=1), "B 信号强度定仓", gB.sum(axis=1)/3))

    # C. both
    tr["w_both"] = (tr.w_risk * tr.w_edge).clip(0.4, 2.5).round(2)
    rC = weighted_bars(tr, bars, "w_both").loc[lambda d: d.index >= start]
    gC = sum(occupancy(g, bars) * m for m, g in tr.groupby("w_both")).loc[lambda d: d.index >= start]
    rows.append(score((rC * (1/3)).sum(axis=1), "C = A×B", gC.sum(axis=1)/3))

    show(rows)

    # Scale every variant to the same 24% annual vol so the comparison is about
    # shape, not size.
    print("\n--- 同波动率 (24%) 对比 ---")
    rows2 = []
    for name, rr in (("基线", r0), ("A 等风险", rA), ("B 信号强度", rB), ("C A×B", rC)):
        p = (rr * (1/3)).sum(axis=1)
        d = (1 + p).resample("1D").prod() - 1
        k = 0.242 / (d.std(ddof=1) * np.sqrt(365))
        rows2.append(score(p * k, f"{name} ×{k:.2f}"))
    show(rows2)

    print("\n--- 基线逐年 ---")
    p = (r0 * (1/3)).sum(axis=1)
    d = (1 + p).resample("1D").prod() - 1
    for y, g in d.groupby(d.index.year):
        print(f"  {y}  收益 {((1+g).prod()-1):>+7.1%}   夏普 "
              f"{g.mean()/g.std(ddof=1)*np.sqrt(365):>+5.2f}   "
              f"回撤 {((1+g).cumprod()/(1+g).cumprod().cummax()-1).min():>+6.1%}")

    print("\n--- 成本敏感 (基线) ---")
    for c in (6.0, 8.0, 10.0, 12.0, 16.0):
        t2 = simulate(pd.read_csv(RES / "oos_dir_c.csv.gz").assign(
            dt=lambda x: pd.to_datetime(x.dt, utc=True)), "c", 0.05, "both", c,
            "model", np.random.default_rng(11))
        rr = bar_returns(t2, bars, cost_bps=c).loc[lambda d_: d_.index >= start]
        s = score((rr * (1/3)).sum(axis=1), f"{c:.0f}bps")
        print(f"  {c:>4.0f}bps  年化 {s['cagr']:>+7.1%}  夏普 {s['sharpe']:>+5.2f}")
