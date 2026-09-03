"""Trade-level P&L for the direction-pure model.

Same machinery as backtest.py -- realised barrier exits, one position per symbol
at a time, controls alongside -- but driven by a two-tailed score: the high tail
of P(up) is a long, the low tail is a short.

The reason this backtest is the arbiter and not the decomposition table: the
decomposition prices only the barrier outcomes, while roughly 58% of trades time
out and exit at market.  If the model has real directional information those
time-outs drift favourably too, and the barrier-only arithmetic understates the
edge.  If it does not, they drift against.  Only the realised P&L knows.

Usage:  python3 -m scripts.backtest.backtest_dir [config] [--no-session]
        [--tails=0.01,0.02]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.backtest.backtest import summarise, COSTS_BPS

RESULTS = Path(__file__).resolve().parents[2] / "results" / "crypto"
ALL_TAILS = (0.10, 0.05, 0.02, 0.01, 0.005, 0.001)


def simulate(df: pd.DataFrame, cfg: str, tail: float, policy: str,
             cost_bps: float, variant: str, rng: np.random.Generator) -> pd.DataFrame:
    """Trade-level P&L.

    Thresholds are taken PER ROW from the row's own fold (`hi_{tail}` / `lo_{tail}`
    are written per fold by train_dir).  Pooling them -- e.g. averaging the six
    folds' cuts into one global number -- lets a later fold's threshold decide an
    earlier fold's selection, which is a mild but real leak and makes the result
    sensitive to folds whose early stopping fired at a different tree count.
    """
    out = []
    for sym, g in df.groupby("symbol", sort=True):
        g = g.sort_values("ts").reset_index(drop=True)
        p = g["p_up"].to_numpy()
        hi = g[f"hi_{tail}"].to_numpy()
        lo = g[f"lo_{tail}"].to_numpy()
        held, exret = g[f"held_{cfg}"].to_numpy(), g[f"exit_ret_{cfg}"].to_numpy()
        want_l, want_s = p >= hi, p <= lo
        if variant == "inverted":
            want_l, want_s = want_s, want_l
        elif variant == "random":
            kl, ks = int(want_l.sum()), int(want_s.sum())
            want_l = np.zeros(len(g), bool); want_s = np.zeros(len(g), bool)
            if kl:
                want_l[rng.choice(len(g), kl, replace=False)] = True
            if ks:
                want_s[rng.choice(len(g), ks, replace=False)] = True
        if policy == "long":
            want_s = np.zeros(len(g), bool)
        elif policy == "short":
            want_l = np.zeros(len(g), bool)
        both = want_l & want_s
        want_l &= ~both
        want_s &= ~both

        free_at = 0
        for i in range(len(g)):
            if i < free_at or not np.isfinite(held[i]) or not np.isfinite(exret[i]):
                continue
            side = 1 if want_l[i] else (-1 if want_s[i] else 0)
            if not side:
                continue
            gross = side * exret[i]
            out.append((sym, g["ts"].iloc[i], g["dt"].iloc[i], side,
                        gross, gross - cost_bps / 1e4, held[i]))
            free_at = i + int(held[i]) + 1
    return pd.DataFrame(out, columns=["symbol", "ts", "dt", "side", "gross", "net", "held"])


def run(cfg: str, tag: str, tails=ALL_TAILS) -> None:
    oos = pd.read_csv(RESULTS / f"oos_dir_{tag}.csv.gz")
    oos["dt"] = pd.to_datetime(oos["dt"], utc=True)
    years = (oos["dt"].max() - oos["dt"].min()).total_seconds() / (365.25 * 86400)
    rng = np.random.default_rng(11)
    print(f"direction backtest  {tag}   {oos['dt'].min():%Y-%m-%d} -> "
          f"{oos['dt'].max():%Y-%m-%d}  ({years:.2f}y)\n")

    rows = []
    for t in tails:
        for policy in ("both", "long", "short"):
            for variant in ("model", "inverted", "random"):
                for cost in COSTS_BPS:
                    s = summarise(simulate(oos, cfg, t, policy, cost, variant, rng), years)
                    s.update(tail=t, policy=policy, variant=variant, cost=cost)
                    rows.append(s)
    res = pd.DataFrame(rows)
    res.to_csv(RESULTS / f"backtest_dir_{tag}.csv", index=False)

    hdr = (f"{'tail':>6} {'policy':>6} {'variant':>9} {'cost':>5} {'trades':>7} {'/mo':>6} "
           f"{'win%':>6} {'gross':>7} {'net':>7} {'t':>6} {'Sharpe':>7} {'tot%':>8} "
           f"{'maxDD':>7} {'hold_h':>7}")
    print(hdr); print("-" * len(hdr))
    for _, r in res.iterrows():
        if not r["trades"]:
            continue
        print(f"{r['tail']:>6.3f} {r['policy']:>6} {r['variant']:>9} {r['cost']:>5.0f} "
              f"{r['trades']:>7.0f} {r['per_month']:>6.1f} {r['winrate']:>5.1%} "
              f"{r['gross_bps']:>+7.1f} {r['net_bps']:>+7.1f} {r['t_stat']:>6.2f} "
              f"{r['sharpe']:>7.2f} {r['total_ret']:>+7.1%} {r['max_dd']:>+6.1%} {r['hold_h']:>7.1f}")

    best = res[(res.variant == "model") & (res.policy == "both") & (res.cost == COSTS_BPS[0])]
    if len(best):
        t = best.sort_values("net_bps").iloc[-1]["tail"]
        tr = simulate(oos, cfg, t, "both", COSTS_BPS[0], "model", rng)
        if not tr.empty:
            print(f"\nbest both-sides cell (tail={t}) by year and symbol")
            tr["year"] = tr["dt"].dt.year
            for key, g in tr.groupby("year"):
                s = summarise(g, 1.0)
                print(f"  {key}     n={s['trades']:>4}  win {s['winrate']:>5.1%}  "
                      f"net {s['net_bps']:>+7.1f}bps  t {s['t_stat']:>+5.2f}")
            for sym, g in tr.groupby("symbol"):
                s = summarise(g, years)
                print(f"  {sym:<9} n={s['trades']:>4}  win {s['winrate']:>5.1%}  "
                      f"net {s['net_bps']:>+7.1f}bps  t {s['t_stat']:>+5.2f}")
            tr.to_csv(RESULTS / f"trades_dir_{tag}.csv", index=False)


if __name__ == "__main__":
    pos = [x for x in sys.argv[1:] if not x.startswith("--")]
    cfg = pos[0] if pos else "a"
    roll = next((x.split("=")[1] for x in sys.argv if x.startswith("--roll=")), None)
    tails_arg = next((x.split("=", 1)[1] for x in sys.argv if x.startswith("--tails=")), None)
    tails = ALL_TAILS if tails_arg is None else tuple(float(x) for x in tails_arg.split(",") if x)
    invalid = [x for x in tails if x not in ALL_TAILS]
    if not tails or invalid:
        raise SystemExit(f"invalid --tails; choose from {','.join(str(x) for x in ALL_TAILS)}")
    run(cfg, f"{cfg}{'_nosession' if '--no-session' in sys.argv else ''}"
             f"{'_okx' if '--okx' in sys.argv else ''}"
             f"{f'_roll{roll}' if roll else ''}", tails=tails)
