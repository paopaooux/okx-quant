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


def _recovery_stats(curve: pd.Series) -> dict:
    """Measure realized-equity drawdown recovery, matching stock reports."""
    if curve.empty:
        return {"max_drawdown_recovery_days": np.nan,
                "longest_drawdown_recovery_days": np.nan,
                "unrecovered_drawdown": False}
    values = curve.sort_index().to_numpy(dtype=float)
    times = curve.sort_index().index
    peaks = np.maximum.accumulate(values)
    active = None
    episodes = []
    for i, value in enumerate(values):
        if active is None and value < peaks[i] * (1 - 1e-12):
            active = {"peak_value": peaks[i], "trough_ts": times[i],
                      "trough_value": value}
        elif active is not None:
            if value < active["trough_value"]:
                active["trough_ts"] = times[i]
                active["trough_value"] = value
            if value >= active["peak_value"] - 1e-8:
                episodes.append({
                    "depth": active["trough_value"] / active["peak_value"] - 1.0,
                    "recovery_days": (times[i] - active["trough_ts"]).total_seconds() / 86400.0,
                })
                active = None
    if active is not None:
        deepest = min(episodes, key=lambda e: e["depth"], default=None)
        return {
            "max_drawdown_recovery_days": float(deepest["recovery_days"]) if deepest else np.nan,
            "longest_drawdown_recovery_days": float(max((e["recovery_days"] for e in episodes), default=np.nan)),
            "unrecovered_drawdown": True,
        }
    deepest = min(episodes, key=lambda e: e["depth"], default=None)
    return {
        "max_drawdown_recovery_days": float(deepest["recovery_days"]) if deepest else np.nan,
        "longest_drawdown_recovery_days": float(max((e["recovery_days"] for e in episodes), default=np.nan)),
        "unrecovered_drawdown": False,
    }


def curve(tr: pd.DataFrame, weight_per_sleeve: float | None = None) -> pd.Series:
    if tr.empty:
        return pd.Series(dtype=float)
    tr = tr.copy()
    symbols = tuple(sorted(tr["symbol"].dropna().astype(str).unique()))
    if weight_per_sleeve is None:
        weight_per_sleeve = 1.0 / len(symbols)
    tr["exit_ms"] = tr["ts"] + BAR_MS * (tr["held"].astype(int) + 1)
    tr["exit_dt"] = pd.to_datetime(tr["exit_ms"], unit="ms", utc=True)
    lo = pd.to_datetime(tr["ts"].min(), unit="ms", utc=True).normalize()
    hi = tr["exit_dt"].max().normalize()
    days = pd.date_range(lo, hi, freq="D", tz="UTC")

    total = pd.Series(0.0, index=days)
    for s in symbols:
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
    symbols = tuple(sorted(tr["symbol"].dropna().astype(str).unique()))
    weight = 1.0 / len(symbols) if symbols else 0.0
    hold_hours = tr["held"].astype(float) * 0.25
    avg_exposure = float((hold_hours * weight).sum() / (yrs * 365.25 * 24.0)) if yrs > 0 else np.nan
    recovery = _recovery_stats(eq)
    return dict(
        years=yrs,
        total_ret=tot,
        cagr=(1 + tot) ** (1 / yrs) - 1 if tot > -1 else np.nan,
        max_dd=float((eq / eq.cummax() - 1).min()),
        sharpe=float(r.mean() / r.std(ddof=1) * np.sqrt(365)) if r.std() else np.nan,
        vol_ann=float(r.std(ddof=1) * np.sqrt(365)),
        days_in_market=float((r != 0).mean()),
        # Canonical names shared with the stock research reports.
        total_return=tot,
        annualized_return=(1 + tot) ** (1 / yrs) - 1 if tot > -1 else np.nan,
        max_drawdown_pct=float((eq / eq.cummax() - 1).min()),
        sharpe_daily=float(r.mean() / r.std(ddof=1) * np.sqrt(365)) if r.std() else np.nan,
        vol_annualized=float(r.std(ddof=1) * np.sqrt(365)),
        span_days=float((eq.index[-1] - eq.index[0]).days),
        n_trades=int(len(tr)),
        avg_hold_hours=float(hold_hours.mean()) if len(hold_hours) else np.nan,
        median_hold_hours=float(hold_hours.median()) if len(hold_hours) else np.nan,
        avg_concurrent=float(hold_hours.sum() / (yrs * 365.25 * 24.0)) if yrs > 0 else np.nan,
        avg_exposure=avg_exposure,
        **recovery,
    )
