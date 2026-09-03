"""The one lever the research never tested: an exit that uses the model.

At config c, 82% of trades exit on the H=48 timeout rather than on a barrier --
i.e. four out of five exits are decided by a clock, not by information.  The
model already scores every bar out of sample, so `p_up` during the holding
window is available and free.  This asks whether letting a long close early when
its own p_up falls back through a level beats holding to the clock.

Entry, cost and barrier logic are untouched; only the exit moves.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from scripts.analysis.opt_probe import load_bars, score, show, SYMS

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results" / "crypto"
COST = 10.0


def run(exit_level: float | None, cfg="c", tail=0.05):
    """exit_level=None reproduces the barrier-only book."""
    oos = pd.read_csv(RES / f"oos_dir_{cfg}.csv.gz")
    oos["dt"] = pd.to_datetime(oos["dt"], utc=True)
    bars = load_bars()
    series, held_h, n = {}, [], 0
    for s in SYMS:
        k = bars[s]
        pos = pd.Index(k["ts"].to_numpy())
        op = k["open"].to_numpy(float)
        g = oos[oos.symbol == s].sort_values("ts").reset_index(drop=True)
        p = g["p_up"].to_numpy()
        hi, lo = g[f"hi_{tail}"].to_numpy(), g[f"lo_{tail}"].to_numpy()
        held = g[f"held_{cfg}"].to_numpy()
        exret = g[f"exit_ret_{cfg}"].to_numpy()
        gts = g["ts"].to_numpy()
        out = np.zeros(len(op))
        free_at = 0
        for i in range(len(g)):
            if i < free_at or not np.isfinite(held[i]) or not np.isfinite(exret[i]):
                continue
            side = 1 if p[i] >= hi[i] else (-1 if p[i] <= lo[i] else 0)
            if not side:
                continue
            e = pos.get_loc(int(gts[i])) + 1        # entry at next bar's open
            h = int(held[i])
            if e + h >= len(op):
                continue
            cut = h
            if exit_level is not None:
                # Scan the holding window on the model's own out-of-sample score.
                for j in range(1, h):
                    if i + j >= len(g):
                        break
                    q = p[i + j]
                    if (side > 0 and q < exit_level) or (side < 0 and q > 1 - exit_level):
                        cut = j
                        break
            path = side * (op[e + 1:e + cut + 1] / op[e:e + cut] - 1.0)
            if cut == h:                            # realised barrier / timeout exit
                resid = (1 + side * float(exret[i])) / float(np.prod(1 + path)) - 1
                seq = np.append(path, resid)
            else:
                seq = path
            seq = seq.copy(); seq[0] -= COST / 1e4
            out[e:e + len(seq)] = seq
            held_h.append(cut * 0.25); n += 1
            free_at = i + cut + 1
        series[s] = pd.Series(out, index=pd.to_datetime(k["ts"], unit="ms", utc=True))
    df = pd.DataFrame(series)
    df = df.loc[df.index >= oos.dt.min()]
    return (df * (1/3)).sum(axis=1), n, float(np.mean(held_h))


if __name__ == "__main__":
    rows = []
    for lvl, name in ((None, "屏障出场（现状）"), (0.50, "p_up 跌破 0.50 就走"),
                      (0.48, "跌破 0.48"), (0.45, "跌破 0.45"), (0.40, "跌破 0.40")):
        p, n, h = run(lvl)
        s = score(p, name)
        s["n"] = n; s["hold_h"] = round(h, 1)
        rows.append(s)
    show(rows)
