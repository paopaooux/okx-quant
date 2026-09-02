"""Hold time, win rate and how long the drawdowns actually last.

Two things the existing tables cannot answer.  First, `max_dd` is a depth; the
question "how long was I underwater" is a duration and nobody measured it.
Second, README section 8 swept the asymmetric exits only on the tail=0.005 cell
(9 trades a month), so its "+11.3% annualised" is mostly a frequency effect, not
a verdict on short holds -- the same exits on the tail=0.05 cell are a different
trade-off entirely.

Everything is scored on the mark-to-market path (see opt_probe.py), and the
random control is rebuilt at every cell because an asymmetric payoff buys a high
win rate whether or not the model knows anything.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.backtest import asym
from scripts.analysis.opt_probe import load_bars, bar_returns, SYMS

BARS = load_bars()


def underwater(daily: pd.Series) -> dict:
    eq = (1 + daily).cumprod()
    peak = eq.cummax()
    under = eq < peak * (1 - 1e-9)
    runs, cur = [], 0
    for flag in under:
        if flag:
            cur += 1
        elif cur:
            runs.append(cur); cur = 0
    if cur:
        runs.append(cur)          # still underwater at the end of the sample
    dd = eq / peak - 1
    trough = int(np.argmin(dd.to_numpy()))
    after = eq.iloc[trough:]
    rec = np.where(after.to_numpy() >= peak.iloc[trough])[0]
    return {
        "maxdd": float(dd.min()),
        "水下最长(天)": max(runs) if runs else 0,
        "水下中位(天)": int(np.median(runs)) if runs else 0,
        "水下时间占比": float(under.mean()),
        "最深回撤恢复(天)": int(rec[0]) if len(rec) else -1,   # -1 = not yet recovered
    }


def path_stats(tr: pd.DataFrame, cost=10.0) -> dict:
    r = bar_returns(tr, BARS, cost_bps=cost)
    r = r.loc[r.index >= tr.dt.min()]
    p = (r * (1 / 3)).sum(axis=1)
    d = (1 + p).resample("1D").prod() - 1
    eq = (1 + d).cumprod()
    yrs = (d.index[-1] - d.index[0]).days / 365.25
    out = {
        "笔数": len(tr),
        "月频": round(len(tr) / (yrs * 12), 1),
        "胜率": float((tr.net > 0).mean()),
        "持仓h": round(float(tr.held.mean()) * 0.25, 1),
        "年化": float(eq.iloc[-1] ** (1 / yrs) - 1),
        "夏普": float(d.mean() / d.std(ddof=1) * np.sqrt(365)),
    }
    out.update(underwater(d))
    return out


def show(rows):
    f = pd.DataFrame(rows).set_index("name")
    for c in ("胜率", "年化", "maxdd", "水下时间占比"):
        f[c] = f[c].map("{:+.1%}".format) if c != "胜率" else f[c].map("{:.1%}".format)
    f["夏普"] = f["夏普"].map("{:+.2f}".format)
    print(f.to_string())


if __name__ == "__main__":
    oos, px = asym.load("c")
    rows = []
    for tail, tp, sl, label in ((0.05, 1.0, 1.0, "现状 对称 tail.05"),
                                (0.05, 0.7, 2.0, "tail.05 TP0.7/SL2.0"),
                                (0.05, 0.6, 1.2, "tail.05 TP0.6/SL1.2"),
                                (0.05, 0.4, 1.0, "tail.05 TP0.4/SL1.0"),
                                (0.02, 0.4, 1.0, "tail.02 TP0.4/SL1.0"),
                                (0.005, 0.4, 1.0, "tail.005 TP0.4/SL1.0")):
        m, rnd = asym.trades(oos, px, "c", tail, "both", tp, sl)
        s = path_stats(m); s["name"] = label
        s["随机胜率"] = f"{(rnd.net > 0).mean():.1%}"
        s["随机净bps"] = round(float(rnd.net.mean() * 1e4), 1)
        rows.append(s)

    oosb, pxb = asym.load("b")
    for tail, tp, sl, label in ((0.05, 0.4, 1.0, "配置b tail.05 TP0.4/SL1.0"),
                                (0.01, 0.5, 2.0, "配置b tail.01 TP0.5/SL2.0")):
        m, rnd = asym.trades(oosb, pxb, "b", tail, "both", tp, sl)
        s = path_stats(m); s["name"] = label
        s["随机胜率"] = f"{(rnd.net > 0).mean():.1%}"
        s["随机净bps"] = round(float(rnd.net.mean() * 1e4), 1)
        rows.append(s)

    show(rows)
