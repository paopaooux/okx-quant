"""Asymmetric take-profit / stop-loss exits on top of the trained direction model.

Why this exists: the objective asks for a HIGH WIN RATE, and the symmetric
1:1 barrier the labels are built on cannot deliver one.  With a 1:1 barrier a
49% win rate and a +30bps mean are the same statement -- the payoff is symmetric
so the hit rate is pinned near the direction edge.  Moving the take-profit
closer than the stop trades payoff for frequency: at TP = p*w and SL = q*w with
p < q, the hit rate rises toward q/(p+q) even with zero edge.

That "even with zero edge" is the trap, and it is why this module reports the
random control at every (p, q): a high win rate bought purely with an asymmetric
payoff is not information, and its expectancy is still negative after cost.  The
number to read is net bps and its t, with the win rate as a description of the
*shape* of the return -- never as evidence.

The model is NOT retrained.  Direction is direction; only the exit changes.  So
the p/q surface is a post-hoc exit sweep on a fixed signal, and the whole surface
is printed rather than its best cell.

Usage: python3 -m scripts.backtest.asym [cfg] [tail] [policy]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.data import build
from scripts.backtest.backtest import summarise
from scripts.backtest import portfolio

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
COST = 10.0
# TP multiple x SL multiple, both in units of the config's symmetric barrier w.
GRID = [(0.4, 1.0), (0.4, 1.5), (0.5, 1.0), (0.5, 1.5), (0.5, 2.0),
        (0.6, 1.2), (0.6, 1.5), (0.7, 1.5), (0.7, 2.0), (1.0, 1.0)]


def asym_exit(open_, high, low, width, horizon, tp_m, sl_m, side, where,
              delay=0, sl_slip_bps=0.0):
    """Gross return and bars held for a `side` trade with asymmetric barriers.

    Entry is the next bar's open, matching build.triple_barrier.  A bar whose
    range spans both barriers is scored as the STOP firing first -- the same
    pessimistic convention the symmetric labels use.

    `delay` postpones the fill by that many extra bars -- the execution-latency
    stress test: 0 is "we get the next open", 1 is "we are a full 15m late".
    `sl_slip_bps` charges extra on a STOP fill only, since a stop-market crosses
    into a moving book while a take-profit rests as a limit and does not.

    `where` restricts the walk to the bars that are actually traded.  Evaluating
    all ~200k bars for every (tp, sl, side, symbol) combination is ~600M inner
    steps of pure Python; the traded set is three orders of magnitude smaller and
    the result is identical.
    """
    n = len(open_)
    ret = np.full(n, np.nan)
    held = np.full(n, np.nan)
    for i in where:
        if i >= n - horizon - 1 - delay:
            continue
        w = width[i]
        entry = open_[i + 1 + delay]
        if not np.isfinite(w) or w <= 0 or not np.isfinite(entry) or entry <= 0:
            continue
        tp_r, sl_r = tp_m * w, sl_m * w
        if side > 0:
            tp_px, sl_px = entry * (1 + tp_r), entry * (1 - sl_r)
        else:
            tp_px, sl_px = entry * (1 - tp_r), entry * (1 + sl_r)
        end = i + 1 + delay + horizon
        out_r = out_h = None
        for j in range(i + 1 + delay, end + 1):
            hit_tp = high[j] >= tp_px if side > 0 else low[j] <= tp_px
            hit_sl = low[j] <= sl_px if side > 0 else high[j] >= sl_px
            if hit_sl:                      # stop wins ties -- pessimistic
                out_r, out_h = -sl_r - sl_slip_bps / 1e4, j - i
                break
            if hit_tp:
                out_r, out_h = tp_r, j - i
                break
        if out_r is None:
            out_r, out_h = side * (open_[end] / entry - 1.0), horizon
        ret[i], held[i] = out_r, out_h
    return ret, held


def load(cfg: str):
    """OOS scores + raw OHLC, shared by every (tp, sl) cell.  Split out of main so
    other modules (report tables) can build the same trade list without re-running
    the whole grid."""
    oos = pd.read_csv(ROOT / "results" / f"oos_dir_{cfg}.csv.gz")
    oos["dt"] = pd.to_datetime(oos["dt"], utc=True)
    px = {}
    for sym in ("BTCUSDT", "ETHUSDT", "SOLUSDT"):
        k = pd.read_csv(DATA / "klines" / f"{sym}.csv.gz",
                        usecols=["ts", "open", "high", "low"]).sort_values("ts")
        px[sym] = k.reset_index(drop=True)
    return oos, px


def trades(oos, px, cfg, tail, pol, tp_m, sl_m, cost=COST,
           delay=0, sl_slip_bps=0.0):
    """Trade list for one (tp, sl) cell.  Returns (model, random)."""
    H = dict((n, h) for n, _, h in build.CONFIGS)[cfg]
    recs, rnd = [], []
    for sym, g in oos.groupby("symbol", sort=True):
        g = g.sort_values("ts").reset_index(drop=True)
        k = px[sym]
        pos = pd.Series(np.arange(len(k)), index=k["ts"].to_numpy())
        idx = g["ts"].map(pos).to_numpy()
        o, h, l = (k["open"].to_numpy(), k["high"].to_numpy(), k["low"].to_numpy())
        w_full = np.full(len(k), np.nan)
        ok = np.isfinite(idx.astype(float))
        w_full[idx[ok].astype(int)] = g[f"width_{cfg}"].to_numpy()[ok]

        p = g["p_up"].to_numpy()
        hi, lo = g[f"hi_{tail}"].to_numpy(), g[f"lo_{tail}"].to_numpy()
        want_l, want_s = p >= hi, p <= lo
        if pol == "long":
            want_s = np.zeros(len(g), bool)
        elif pol == "short":
            want_l = np.zeros(len(g), bool)
        both = want_l & want_s
        want_l &= ~both; want_s &= ~both

        rng = np.random.default_rng(7)
        kl, ks = int(want_l.sum()), int(want_s.sum())
        rnd_l = np.zeros(len(g), bool); rnd_s = np.zeros(len(g), bool)
        if kl:
            rnd_l[rng.choice(len(g), kl, replace=False)] = True
        if ks:
            rnd_s[rng.choice(len(g), ks, replace=False)] = True
        rnd_s &= ~rnd_l

        need_l = np.unique(idx[(want_l | rnd_l)].astype(int))
        need_s = np.unique(idx[(want_s | rnd_s)].astype(int))
        cache = {1: asym_exit(o, h, l, w_full, H, tp_m, sl_m, 1, need_l,
                              delay, sl_slip_bps),
                 -1: asym_exit(o, h, l, w_full, H, tp_m, sl_m, -1, need_s,
                               delay, sl_slip_bps)}

        for tag, sel_l, sel_s in (("model", want_l, want_s),
                                  ("random", rnd_l, rnd_s)):
            free_at = 0
            dst = recs if tag == "model" else rnd
            for i in range(len(g)):
                if i < free_at:
                    continue
                side = 1 if sel_l[i] else (-1 if sel_s[i] else 0)
                if not side:
                    continue
                j = int(idx[i])
                r, hh = cache[side][0][j], cache[side][1][j]
                if not np.isfinite(r):
                    continue
                dst.append((sym, g["ts"].iloc[i], g["dt"].iloc[i], side,
                            r, r - cost / 1e4, hh))
                free_at = i + int(hh) + 1

    cols = ["symbol", "ts", "dt", "side", "gross", "net", "held"]
    return pd.DataFrame(recs, columns=cols), pd.DataFrame(rnd, columns=cols)


def main() -> None:
    cfg = sys.argv[1] if len(sys.argv) > 1 else "c"
    tail = float(sys.argv[2]) if len(sys.argv) > 2 else 0.02
    pol = sys.argv[3] if len(sys.argv) > 3 else "both"
    H = dict((n, h) for n, _, h in build.CONFIGS)[cfg]

    oos = pd.read_csv(ROOT / "results" / f"oos_dir_{cfg}.csv.gz")
    oos["dt"] = pd.to_datetime(oos["dt"], utc=True)
    yrs = (oos.dt.max() - oos.dt.min()).total_seconds() / (365.25 * 86400)

    px = {}
    for sym in ("BTCUSDT", "ETHUSDT", "SOLUSDT"):
        k = pd.read_csv(DATA / "klines" / f"{sym}.csv.gz",
                        usecols=["ts", "open", "high", "low"]).sort_values("ts")
        px[sym] = k.reset_index(drop=True)

    print(f"非对称出场扫描  cfg={cfg}  H={H}  tail={tail}  {pol}  cost={COST}bps")
    print(f"OOS {oos.dt.min():%Y-%m} -> {oos.dt.max():%Y-%m} ({yrs:.2f}y)\n")
    hdr = (f"{'TP':>5} {'SL':>5} | {'n':>5} {'/mo':>5} {'win%':>6} {'net':>7} {'t':>6} "
           f"{'hold_h':>7} | {'年化':>7} {'回撤':>7} {'夏普':>6} | {'随机win%':>9} {'随机net':>8}")
    print(hdr); print("-" * len(hdr))

    rows = []
    for tp_m, sl_m in GRID:
        m, r = trades(oos, px, cfg, tail, pol, tp_m, sl_m)
        if m.empty:
            continue
        sm, sr = summarise(m, yrs), summarise(r, yrs) if not r.empty else {}
        pf = portfolio.stats(m)
        print(f"{tp_m:>5.1f} {sl_m:>5.1f} | {sm['trades']:>5.0f} {sm['per_month']:>5.1f} "
              f"{sm['winrate']:>5.1%} {sm['net_bps']:>+7.1f} {sm['t_stat']:>+6.2f} "
              f"{sm['hold_h']:>7.1f} | {pf.get('cagr', float('nan')):>+6.1%} "
              f"{pf.get('max_dd', float('nan')):>+6.1%} {pf.get('sharpe', float('nan')):>6.2f} "
              f"| {sr.get('winrate', float('nan')):>8.1%} {sr.get('net_bps', float('nan')):>+8.1f}")
        rows.append({"tp": tp_m, "sl": sl_m, **sm,
                     **{f"pf_{k}": v for k, v in pf.items()}})
    pd.DataFrame(rows).to_csv(ROOT / "results" / f"asym_{cfg}_{tail}_{pol}.csv", index=False)


if __name__ == "__main__":
    main()
