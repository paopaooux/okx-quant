"""Where the remaining optimisation room is, measured rather than guessed.

The README's annualised / drawdown / Sharpe come from `portfolio.py`, which
books a trade's whole P&L on its exit bar.  That is fine for scoring a fixed
book, but it cannot answer "how much capital was actually working", because it
never represents the holding period as a path.  Every question about sizing --
sleeve weights, pooling idle capital, running two configs at once -- is a
question about the path, so this rebuilds one.

Per-bar path: open-to-open returns over the holding window, with the final bar
forced so the trade's compounded total equals the realised `exit_ret` from the
backtest.  Barrier exits happen intrabar, so the last bar is where the residual
has to go; everything before it is the real price path.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from scripts.backtest.backtest_dir import simulate

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results" / "crypto"
SYMS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
BAR_MS = 15 * 60 * 1000
BARS_Y = 365 * 96


def load_bars() -> dict[str, pd.DataFrame]:
    out = {}
    for s in SYMS:
        k = pd.read_csv(ROOT / "data" / "klines" / f"{s}.csv.gz")
        k = k.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
        out[s] = k
    return out


def trades(tag: str, cfg: str, tail: float, cost: float = 10.0) -> pd.DataFrame:
    oos = pd.read_csv(RES / f"oos_dir_{tag}.csv.gz")
    oos["dt"] = pd.to_datetime(oos["dt"], utc=True)
    rng = np.random.default_rng(11)
    return simulate(oos, cfg, tail, "both", cost, "model", rng)


def bar_returns(tr: pd.DataFrame, bars: dict[str, pd.DataFrame],
                cost_bps: float = 10.0) -> pd.DataFrame:
    """One column per symbol of per-bar *unit-notional* net return, 0 when flat."""
    grids = {}
    for s in SYMS:
        k = bars[s]
        ts = k["ts"].to_numpy()
        op = k["open"].to_numpy(float)
        pos = pd.Index(ts)
        r = np.zeros(len(ts))
        g = tr[tr.symbol == s]
        for _, row in g.iterrows():
            i = pos.get_loc(int(row["ts"]))
            e = i + 1                      # entry is the NEXT bar's open
            h = int(row["held"])
            if e + h >= len(ts):
                continue
            path = op[e + 1:e + h + 1] / op[e:e + h] - 1.0
            path = row["side"] * path
            # Force the compounded total onto the realised barrier exit.
            done = float(np.prod(1 + path))
            resid = (1 + float(row["gross"])) / done - 1
            seq = np.append(path, resid)
            seq[0] -= cost_bps / 1e4       # both legs of cost at entry
            r[e:e + h + 1] = seq
        grids[s] = pd.Series(r, index=pd.to_datetime(ts, unit="ms", utc=True))
    return pd.DataFrame(grids)


def occupancy(tr: pd.DataFrame, bars: dict[str, pd.DataFrame]) -> pd.DataFrame:
    grids = {}
    for s in SYMS:
        ts = bars[s]["ts"].to_numpy()
        pos = pd.Index(ts)
        occ = np.zeros(len(ts))
        for _, row in tr[tr.symbol == s].iterrows():
            i = pos.get_loc(int(row["ts"])); e = i + 1; h = int(row["held"])
            occ[e:e + h + 1] = 1.0
        grids[s] = pd.Series(occ, index=pd.to_datetime(ts, unit="ms", utc=True))
    return pd.DataFrame(grids)


def score(port: pd.Series, label: str, gross: pd.Series | None = None) -> dict:
    d = (1 + port).resample("1D").prod() - 1
    d = d[d.index >= port.index[0].normalize()]
    eq = (1 + d).cumprod()
    yrs = (d.index[-1] - d.index[0]).days / 365.25
    tot = eq.iloc[-1] - 1
    out = dict(
        name=label,
        cagr=(1 + tot) ** (1 / yrs) - 1,
        maxdd=float((eq / eq.cummax() - 1).min()),
        sharpe=float(d.mean() / d.std(ddof=1) * np.sqrt(365)),
        vol=float(d.std(ddof=1) * np.sqrt(365)),
    )
    if gross is not None:
        out["gross_avg"] = float(gross.mean())
        out["gross_p99"] = float(gross.quantile(0.99))
        out["gross_max"] = float(gross.max())
    return out


def show(rows: list[dict]) -> None:
    f = pd.DataFrame(rows).set_index("name")
    for c in ("cagr", "maxdd", "vol", "gross_avg", "gross_p99", "gross_max"):
        if c in f:
            f[c] = f[c].map(lambda x: f"{x:+.1%}" if pd.notna(x) else "")
    f["sharpe"] = f["sharpe"].map(lambda x: f"{x:+.2f}")
    print(f.to_string())


if __name__ == "__main__":
    bars = load_bars()
    tr = trades("c", "c", 0.05)
    print(f"config c  tail 0.05  both sides  n={len(tr)}  "
          f"{tr.dt.min():%Y-%m-%d} -> {tr.dt.max():%Y-%m-%d}")

    r = bar_returns(tr, bars)
    occ = occupancy(tr, bars)
    r = r.loc[r.index >= tr.dt.min()]
    occ = occ.loc[occ.index >= tr.dt.min()]

    print("\n--- 资金利用率 ---")
    for s in SYMS:
        print(f"  {s:<9} 在场时间 {occ[s].mean():6.1%}   笔数 {(tr.symbol==s).sum():>4}")
    conc = occ.sum(axis=1)
    print(f"  同时持仓数分布: " + "  ".join(
        f"{k}币={v:.1%}" for k, v in conc.value_counts(normalize=True).sort_index().items()))
    print(f"  平均总仓位(1/3 权重) = {conc.mean()/3:.1%}")

    rows = []
    base = (r * (1 / 3)).sum(axis=1)
    rows.append(score(base, "基线 1/3×3", conc / 3))

    # Lever 1 -- sleeve weights.  BTC is the weak leg (README 10.1); this is an
    # in-sample choice and is labelled as one.
    for name, w in (("去掉BTC 1/2×2", {"BTCUSDT": 0, "ETHUSDT": .5, "SOLUSDT": .5}),
                    ("BTC半仓", {"BTCUSDT": 1/6, "ETHUSDT": 5/12, "SOLUSDT": 5/12})):
        w = pd.Series(w)
        rows.append(score((r * w).sum(axis=1), name, (occ * w).sum(axis=1)))

    # Lever 2 -- pool the idle capital.  Same trades, notional per trade raised
    # so total gross approaches 1.0 when anything is open.
    for f in (0.5, 1.0):
        g = (occ * f).sum(axis=1)
        cap = (1.0 / g.replace(0, np.nan)).clip(upper=1.0).fillna(1.0)
        rows.append(score((r * f).sum(axis=1) * cap, f"每笔{f:.0%}仓位(封顶1x)",
                          (g * cap)))

    show(rows)

    # Lever 3 -- two configs at once.
    trb = trades("b", "b", 0.05)
    rb = bar_returns(trb, bars); ob = occupancy(trb, bars)
    idx = r.index
    rb = rb.reindex(idx).fillna(0.0); ob = ob.reindex(idx).fillna(0.0)
    print(f"\n--- 配置组合 (b tail .05, n={len(trb)}) ---")
    ens_r = (r + rb) / 2
    ens_o = (occ + ob) / 2
    rows2 = [score(base, "只跑 c", conc / 3),
             score((rb * (1/3)).sum(axis=1), "只跑 b", ob.sum(axis=1) / 3),
             score((ens_r * (1/3)).sum(axis=1), "b+c 各半", ens_o.sum(axis=1) / 3)]
    show(rows2)
