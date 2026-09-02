"""What does a round trip ACTUALLY cost on OKX, and does the edge survive it?

`backtest.COSTS_BPS = (10.0, 16.0)` was an assumption written as a comment, not a
measurement.  It is also wrong for the venue that matters: OKX Lv1 taker is
0.05%/side, so TAKER FEES ALONE are 10bps round trip -- the whole budget, with
nothing left for spread, impact or funding.

This module decomposes the real number:
  1. exchange fees, under three execution schemes
  2. funding actually paid/received over the realised holding periods
  3. half-spread paid crossing, from OKX tick sizes
  4. market impact, from per-trade notional vs the bar's traded volume
and then re-runs the headline cells at the total.
"""
import numpy as np, pandas as pd
from pathlib import Path
from scripts.backtest import backtest, backtest_dir as bd, portfolio as pf, asym
from scripts.analysis.funding_impact import funding_bps

ROOT = Path(__file__).resolve().parents[2]

YRS = 3.39
# OKX perpetual swap, regular user Lv1 (okx.com/fees, verified 2026-08).
TAKER, MAKER = 5.0, 2.0
# Tick size / typical price over the sample -> one tick in bps.  A resting book on
# a major perp is one tick wide for most of the day, so half-spread ~ half a tick.
TICK = {"BTCUSDT": (0.1, 60000.), "ETHUSDT": (0.01, 2600.), "SOLUSDT": (0.001, 120.)}


def spread_bps(sym):
    t, p = TICK[sym]
    return t / p * 1e4 / 2          # half-spread, one side


def decompose(tr, tp_frac=None):
    """Per-trade cost breakdown in bps, round trip."""
    out = {}
    # -- fees --------------------------------------------------------------
    out["fee_all_taker"] = 2 * TAKER
    if tp_frac is not None:
        # entry crosses (we act on the bar's open); TP can rest as a limit (maker),
        # SL and timeout must cross.
        out["fee_mixed"] = TAKER + tp_frac * MAKER + (1 - tp_frac) * TAKER
    # -- spread ------------------------------------------------------------
    w = tr.groupby("symbol").size() / len(tr)
    out["spread"] = float(sum(w.get(s, 0) * spread_bps(s) * 2 for s in TICK))
    # -- funding -----------------------------------------------------------
    f = np.array([funding_bps(r.symbol, r.ts, r.held, r.side)
                  for r in tr.itertuples()])
    out["funding_mean"] = -float(f.mean())      # cost sign: positive = we pay
    out["funding_sd"] = float(f.std(ddof=1))
    return out


def capacity(tr, capital_usd):
    """Per-trade notional as a share of the entry bar's traded volume."""
    qv = {}
    for s in TICK:
        k = pd.read_csv(ROOT / "data" / "klines" / f"{s}.csv.gz", usecols=["ts", "quote_volume"])
        qv[s] = pd.Series(k["quote_volume"].to_numpy(), index=k["ts"].to_numpy())
    notional = capital_usd / 3.0                 # one sleeve, 1x
    share = np.array([notional / qv[r.symbol].get(r.ts, np.nan) for r in tr.itertuples()])
    share = share[np.isfinite(share)]
    return notional, np.nanmedian(share) * 100, np.nanpercentile(share, 95) * 100


def main():
    oos, px = asym.load("c")
    print("OKX 永续 Lv1: taker 5.0bps/边  maker 2.0bps/边\n")

    print("=== 1. 半价差（按 tick size / 均价，单边）===")
    for s in TICK:
        print(f"  {s:9} tick {TICK[s][0]:<7} ~{TICK[s][1]:>7.0f} USD -> {spread_bps(s):.3f} bps")

    cells = [("对称 1:1  tail=.02", 0.02, None, None, None),
             ("对称 1:1  tail=.005", 0.005, None, None, None),
             ("TP.4/SL1  tail=.02", 0.02, 0.4, 1.0, None),
             ("TP.4/SL1  tail=.005", 0.005, 0.4, 1.0, None)]

    print("\n=== 2. 每笔成本分解（bps，往返）===")
    print(f"{'格子':<22}{'笔数':>6}{'全taker':>9}{'混合':>7}{'价差':>7}{'资金费':>8}{'±sd':>7}{'合计(全taker)':>14}{'合计(混合)':>12}")
    built = {}
    for label, tail, tp, sl, _ in cells:
        if tp is None:
            tr = bd.simulate(oos, "c", tail, "both", 0.0, "model", np.random.default_rng(11))
            tpf = None
        else:
            tr, _r = asym.trades(oos, px, "c", tail, "both", tp, sl, cost=0.0)
            # share of exits that hit TP: gross > 0 means the near barrier fired
            tpf = float((tr["gross"] > 0).mean())
        d = decompose(tr, tpf)
        tot_t = d["fee_all_taker"] + d["spread"] + d["funding_mean"]
        tot_m = (d.get("fee_mixed", d["fee_all_taker"]) + d["spread"] + d["funding_mean"])
        built[label] = (tr, tpf, d, tot_t, tot_m)
        fm = f"{d['fee_mixed']:>7.1f}" if "fee_mixed" in d else f"{'-':>7}"
        print(f"{label:<22}{len(tr):>6}{d['fee_all_taker']:>9.1f}{fm}{d['spread']:>7.2f}"
              f"{d['funding_mean']:>+8.2f}{d['funding_sd']:>7.2f}{tot_t:>14.1f}{tot_m:>12.1f}")

    print("\n=== 3. 容量：单笔名义 / 该 15m bar 成交额 ===")
    tr = built["对称 1:1  tail=.02"][0]
    print(f"{'账户规模':>12}{'单笔名义':>12}{'中位占比':>10}{'p95 占比':>10}")
    for cap in (1e5, 1e6, 1e7, 5e7):
        n, med, p95 = capacity(tr, cap)
        print(f"{cap:>12,.0f}{n:>12,.0f}{med:>9.3f}%{p95:>9.3f}%")

    print("\n=== 4. 在真实成本下重跑 ===")
    print(f"{'格子':<22}{'成本':>7}{'笔数':>6}{'胜率':>7}{'净bps':>8}{'t':>7}{'年化':>8}{'回撤':>8}{'夏普':>7}")
    for label, tail, tp, sl, _ in cells:
        tr0, tpf, d, tot_t, tot_m = built[label]
        for cname, c in (("混合", tot_m), ("全taker", tot_t)):
            if tp is None:
                tr = bd.simulate(oos, "c", tail, "both", c, "model", np.random.default_rng(11))
            else:
                tr, _r = asym.trades(oos, px, "c", tail, "both", tp, sl, cost=c)
            s, p = backtest.summarise(tr, YRS), pf.stats(tr)
            print(f"{label:<22}{cname:>6}{c:>5.1f}{s['trades']:>6}{s['winrate']*100:>6.1f}%"
                  f"{s['net_bps']:>+8.1f}{s['t_stat']:>+7.2f}{p['cagr']*100:>+7.1f}%"
                  f"{p['max_dd']*100:>+7.1f}%{p['sharpe']:>7.2f}")


if __name__ == "__main__":
    main()
