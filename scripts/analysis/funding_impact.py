"""Apply per-trade OKX funding to the direction-model trades.

Why: positions are held 3-11 hours, spanning one or two 8-hour settlements, so
funding is first-order at this horizon.  It is also the one input that does NOT
transfer from Binance -- the two venues' rates differ, and unlike price
(correlation 0.99911) funding correlates only 0.67-0.90 across venues.

OKX's own /public/funding-rate-history retains ~3 months, so the 3-year series is
Binance's, shifted by the OKX-minus-Binance mean gap measured on the 157
overlapping settlements.  That is an estimate, and its residual std (~0.33bps per
settlement) is carried in the reported spread rather than hidden.

Sign: a SHORT receives funding when the rate is positive.  Binance funding was
positive 85% of the sample, so omitting funding has been working against a
short-biased strategy, not for it.
"""
import pathlib
import numpy as np, pandas as pd
from scripts.backtest import backtest_dir as bd

F = pathlib.Path(__file__).resolve().parents[2] / "data" / "funding"
GAP = {"BTCUSDT": -0.106e-4, "ETHUSDT": -0.083e-4, "SOLUSDT": +0.047e-4}
BAR_MS = 15 * 60 * 1000
CELLS = (("a", "a", 0.02), ("b", "b", 0.01), ("c", "c", 0.01))

fund = {}
for s in GAP:
    f = pd.read_csv(F / f"{s}_bn.csv")
    f["r"] = f["rate"] + GAP[s]
    fund[s] = (f["ts"].to_numpy(), f["r"].to_numpy())


def funding_bps(sym: str, ts: int, held: float, side: int) -> float:
    """Funding credit in bps: settlements crossed in (entry, exit], signed."""
    v, r = fund[sym]
    t0 = ts + BAR_MS                          # entry is the next bar's open
    t1 = ts + BAR_MS * (int(held) + 1)
    tot = r[np.searchsorted(v, t0, "right"):np.searchsorted(v, t1, "right")].sum()
    return -side * tot * 1e4                  # short (side=-1) receives when positive


def main() -> None:
    rng = np.random.default_rng(11)
    for tag, cfg, tl in CELLS:
        oos = pd.read_csv(F.parent.parent / "results" / "crypto" / f"oos_dir_{tag}.csv.gz")
        for pol in ("long", "short"):
            tr = bd.simulate(oos, cfg, tl, pol, 0.0, "model", rng)
            if tr.empty:
                continue
            tr["fund"] = [funding_bps(r.symbol, r.ts, r.held, r.side)
                          for r in tr.itertuples()]
            tr["g"] = tr["gross"] * 1e4
            n = len(tr)
            print(f"cfg {cfg} tail={tl} {pol:5s}  n={n}  持仓 {tr.held.mean()*15/60:.1f}h "
                  f"(≈{tr.held.mean()*15/60/8:.2f} 次结算, {tr.fund.ne(0).mean()*100:.0f}% 的单子跨到结算)")
            print(f"   funding {tr.fund.mean():+.2f}bps/笔  median {tr.fund.median():+.2f}  "
                  f"std {tr.fund.std():.2f}  p5 {tr.fund.quantile(.05):+.2f}  p95 {tr.fund.quantile(.95):+.2f}")
            for cost in (10, 16):
                a = tr.g - cost
                b = a + tr.fund
                ta = a.mean() / a.std(ddof=1) * np.sqrt(n)
                tb = b.mean() / b.std(ddof=1) * np.sqrt(n)
                print(f"   cost={cost:2d}bps  无 funding {a.mean():+6.2f}bps t={ta:+5.2f}"
                      f"   含 funding {b.mean():+6.2f}bps t={tb:+5.2f}")
            print()


if __name__ == "__main__":
    main()
