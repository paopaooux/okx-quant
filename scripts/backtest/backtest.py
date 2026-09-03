"""Trade-level backtest on the out-of-sample scores.

The exit is not re-simulated.  build.py's triple barrier already recorded, for a
trade opened at each bar, which barrier fired and how many bars it took, so the
backtest reads `exit_ret_*` and `held_*` directly.  Label and P&L therefore
cannot disagree -- a common way for this kind of study to flatter itself is to
label on one exit rule and settle on another.

Overlap is the thing that turns a mediocre signal into a fake one.  A high score
persists for many consecutive bars, and counting each as a trade multiplies both
the trade count and the apparent significance.  Here a position blocks new
entries in the same symbol until it closes, which is also what "few trades"
means operationally.

Every row is printed.  No cell is dropped for being unflattering.

Usage:  python3 -m scripts.backtest.backtest [config]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.data import build

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "crypto"
QUANTILES = (0.90, 0.95, 0.98, 0.99, 0.995, 0.999)
COSTS_BPS = (10.0, 16.0)          # base and pessimistic round trip
BARS_PER_YEAR = 365 * 24 * 4


def simulate(df: pd.DataFrame, cfg: str, cut_long: float, cut_short: float,
             policy: str, cost_bps: float, variant: str,
             rng: np.random.Generator) -> pd.DataFrame:
    """One sleeve per symbol; positions block re-entry until closed."""
    out = []
    for sym, g in df.groupby("symbol", sort=True):
        g = g.sort_values("ts").reset_index(drop=True)
        pl, ps = g["p_long"].to_numpy(), g["p_short"].to_numpy()
        held = g[f"held_{cfg}"].to_numpy()
        exret = g[f"exit_ret_{cfg}"].to_numpy()

        want_l = pl >= cut_long
        want_s = ps >= cut_short
        if variant == "inverted":
            want_l, want_s = ps >= cut_short, pl >= cut_long   # trade the other side
        elif variant == "random":
            k_l, k_s = int(want_l.sum()), int(want_s.sum())
            want_l = np.zeros(len(g), bool); want_s = np.zeros(len(g), bool)
            if k_l:
                want_l[rng.choice(len(g), k_l, replace=False)] = True
            if k_s:
                want_s[rng.choice(len(g), k_s, replace=False)] = True

        if policy == "long":
            want_s = np.zeros(len(g), bool)
        elif policy == "short":
            want_l = np.zeros(len(g), bool)
        # A bar that fires both ways carries no direction; skip it rather than
        # letting an arbitrary tie-break decide.
        both = want_l & want_s
        want_l &= ~both
        want_s &= ~both

        free_at = 0
        for i in range(len(g)):
            if i < free_at or not np.isfinite(held[i]) or not np.isfinite(exret[i]):
                continue
            if want_l[i]:
                side = 1
            elif want_s[i]:
                side = -1
            else:
                continue
            gross = side * exret[i]
            out.append((sym, g["ts"].iloc[i], g["dt"].iloc[i], side,
                        gross, gross - cost_bps / 1e4, held[i]))
            free_at = i + int(held[i]) + 1
    return pd.DataFrame(out, columns=["symbol", "ts", "dt", "side",
                                      "gross", "net", "held"])


def summarise(tr: pd.DataFrame, years: float) -> dict:
    if tr.empty:
        return dict(trades=0)
    n = len(tr)
    net = tr["net"].to_numpy()
    per_year = n / years
    eq = np.cumprod(1 + net)
    dd = float((eq / np.maximum.accumulate(eq) - 1).min())
    sd = net.std(ddof=1)
    return dict(
        trades=n,
        per_month=n / (years * 12),
        winrate=float((net > 0).mean()),
        gross_bps=float(tr["gross"].mean() * 1e4),
        net_bps=float(net.mean() * 1e4),
        t_stat=float(net.mean() / (sd / np.sqrt(n))) if sd else np.nan,
        sharpe=float(net.mean() / sd * np.sqrt(per_year)) if sd else np.nan,
        total_ret=float(eq[-1] - 1),
        max_dd=dd,
        hold_h=float(tr["held"].mean() * 0.25),
    )


def run(cfg: str) -> None:
    oos = pd.read_csv(RESULTS / f"oos_{cfg}.csv.gz")
    oos["dt"] = pd.to_datetime(oos["dt"], utc=True)
    years = (oos["dt"].max() - oos["dt"].min()).total_seconds() / (365.25 * 86400)
    rng = np.random.default_rng(11)
    print(f"config {cfg}   out-of-sample window "
          f"{oos['dt'].min():%Y-%m-%d} -> {oos['dt'].max():%Y-%m-%d}  ({years:.2f}y)\n")

    rows = []
    for q in QUANTILES:
        cl = oos[f"cut_long_{q}"].mean()
        cs = oos[f"cut_short_{q}"].mean()
        for policy in ("both", "long", "short"):
            for variant in ("model", "inverted", "random"):
                for cost in COSTS_BPS:
                    tr = simulate(oos, cfg, cl, cs, policy, cost, variant, rng)
                    s = summarise(tr, years)
                    s.update(q=q, policy=policy, variant=variant, cost=cost)
                    rows.append(s)
    res = pd.DataFrame(rows)
    res.to_csv(RESULTS / f"backtest_{cfg}.csv", index=False)

    hdr = (f"{'q':>6} {'policy':>6} {'variant':>9} {'cost':>5} {'trades':>7} "
           f"{'/mo':>6} {'win%':>6} {'gross':>7} {'net':>7} {'t':>6} {'Sharpe':>7} "
           f"{'tot%':>7} {'maxDD':>7} {'hold_h':>7}")
    print(hdr)
    print("-" * len(hdr))
    for _, r in res.iterrows():
        if not r["trades"]:
            continue
        print(f"{r['q']:>6.3f} {r['policy']:>6} {r['variant']:>9} {r['cost']:>5.0f} "
              f"{r['trades']:>7.0f} {r['per_month']:>6.1f} {r['winrate']:>5.1%} "
              f"{r['gross_bps']:>+7.1f} {r['net_bps']:>+7.1f} {r['t_stat']:>6.2f} "
              f"{r['sharpe']:>7.2f} {r['total_ret']:>+6.1%} {r['max_dd']:>+6.1%} "
              f"{r['hold_h']:>7.1f}")

    # Yearly breakdown of the highest-selectivity model cell, both sides, base cost.
    q = QUANTILES[-1]
    tr = simulate(oos, cfg, oos[f"cut_long_{q}"].mean(), oos[f"cut_short_{q}"].mean(),
                  "both", COSTS_BPS[0], "model", rng)
    if not tr.empty:
        print(f"\nq={q} both-sides model, by year and symbol")
        tr["year"] = tr["dt"].dt.year
        for keys, g in tr.groupby(["year"]):
            s = summarise(g, max(0.05, (g['dt'].max()-g['dt'].min()).total_seconds()/(365.25*86400)))
            print(f"  {keys[0]}  n={s['trades']:>4}  win {s['winrate']:>5.1%}  "
                  f"net {s['net_bps']:>+7.1f}bps  t {s['t_stat']:>+5.2f}")
        for sym, g in tr.groupby("symbol"):
            s = summarise(g, years)
            print(f"  {sym:<8} n={s['trades']:>4}  win {s['winrate']:>5.1%}  "
                  f"net {s['net_bps']:>+7.1f}bps  t {s['t_stat']:>+5.2f}")
        tr.to_csv(RESULTS / f"trades_{cfg}_q{q}.csv", index=False)


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "a")
