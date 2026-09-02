"""Portfolio-level annualised return, max drawdown and Sharpe.

`backtest.summarise` compounds the trade list in list order, which is BTC's trades
end-to-end, then ETH's, then SOL's.  That is not an equity curve: the drawdown it
reports is an artefact of concatenation, and the Sharpe is per-trade scaled by
trade frequency rather than a time-based Sharpe.  Those three numbers are exactly
the ones a sizing decision rests on, so they are rebuilt here properly.

Model: capital is split equally across the three symbols; each sleeve holds at most
one position at a time (already enforced by the non-overlap rule) at full sleeve
notional, i.e. 1x per sleeve and up to 1x gross at portfolio level.  A trade's P&L
is booked to the sleeve on its EXIT bar, sleeves compound independently, and the
portfolio curve is their weighted sum on a daily grid.  Sharpe is from daily
portfolio returns, annualised by sqrt(365) (crypto trades every day).
"""
import numpy as np, pandas as pd

BAR_MS = 15 * 60 * 1000
SYMS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")


def curve(tr: pd.DataFrame, weight_per_sleeve: float = 1 / 3) -> pd.Series:
    if tr.empty:
        return pd.Series(dtype=float)
    tr = tr.copy()
    tr["exit_ms"] = tr["ts"] + BAR_MS * (tr["held"].astype(int) + 1)
    tr["exit_dt"] = pd.to_datetime(tr["exit_ms"], unit="ms", utc=True)
    lo = pd.to_datetime(tr["ts"].min(), unit="ms", utc=True).normalize()
    hi = tr["exit_dt"].max().normalize()
    days = pd.date_range(lo, hi, freq="D", tz="UTC")

    total = pd.Series(0.0, index=days)
    for s in SYMS:
        g = tr[tr.symbol == s].sort_values("exit_dt")
        eq = pd.Series(1.0, index=days)
        if len(g):
            e = pd.Series(np.cumprod(1 + g["net"].to_numpy()), index=g["exit_dt"])
            e = e[~e.index.duplicated(keep="last")]
            eq = e.reindex(days.union(e.index)).ffill().fillna(1.0).reindex(days)
        total += weight_per_sleeve * eq
    return total


def stats(tr: pd.DataFrame) -> dict:
    eq = curve(tr)
    if eq.empty or len(eq) < 30:
        return {}
    r = eq.pct_change().dropna()
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    tot = eq.iloc[-1] / eq.iloc[0] - 1
    return dict(
        years=yrs,
        total_ret=tot,
        cagr=(1 + tot) ** (1 / yrs) - 1 if tot > -1 else np.nan,
        max_dd=float((eq / eq.cummax() - 1).min()),
        sharpe=float(r.mean() / r.std(ddof=1) * np.sqrt(365)) if r.std() else np.nan,
        vol_ann=float(r.std(ddof=1) * np.sqrt(365)),
        days_in_market=float((r != 0).mean()),
    )
