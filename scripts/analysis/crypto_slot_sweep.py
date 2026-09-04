"""Compare fixed and range-adaptive portfolio slots for the crypto OOS file.

The capacity rule only uses entry-time barrier width. It never ranks symbols
with realized returns. The live venue is one-way/net, so each symbol has one
position at a time; the standalone crypto audit keeps its original 1/8 risk
budget.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.backtest import backtest_dir as bd
from scripts.backtest.portfolio import _recovery_stats

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "crypto"
BAR_MS = 15 * 60 * 1000


def _strength(frame: pd.DataFrame) -> pd.Series:
    return np.where(
        frame.side > 0,
        (frame.p_up - frame["hi_0.01"]) / np.maximum(1 - frame["hi_0.01"], 1e-9),
        (frame["lo_0.01"] - frame.p_up) / np.maximum(frame["lo_0.01"], 1e-9),
    )


def select_slots(candidates: pd.DataFrame, mode: str) -> tuple[pd.DataFrame, int]:
    """Select entries with a fixed or range-adaptive concurrent capacity."""
    c = candidates.sort_values(["ts", "symbol"]).reset_index(drop=True).copy()
    # Range is known at entry. Quantiles are expanding and use no realized P&L.
    ranges = c.groupby("ts")["width_c"].median().sort_index()
    active: dict[str, int] = {}
    accepted: list[int] = []
    skipped = 0
    history: list[float] = []
    for stamp, group in c.groupby("ts", sort=True):
        stamp = int(stamp)
        active = {s: e for s, e in active.items() if e > stamp}
        value = float(ranges.loc[stamp])
        if mode == "dynamic":
            q25, q75 = np.quantile(history, (0.25, 0.75)) if len(history) >= 20 else (value, value)
            cap = 7 if value >= q75 and len(history) >= 20 else (3 if value <= q25 and len(history) >= 20 else 5)
        else:
            cap = int(mode)
        available = group[
            ~group.apply(lambda row: str(row.symbol) in active, axis=1)
        ]
        chosen = available.sort_values(["strength", "symbol"], ascending=[False, True]).head(max(0, cap - len(active)))
        accepted.extend(chosen.index.tolist())
        skipped += len(available) - len(chosen)
        for _, row in chosen.iterrows():
            active[str(row.symbol)] = stamp + (int(row.held) + 1) * BAR_MS
        history.append(value)
    return c.loc[accepted].sort_values("ts").reset_index(drop=True), skipped


def shared_curve(trades: pd.DataFrame, weight: float) -> pd.Series:
    if trades.empty:
        return pd.Series(dtype=float)
    t = trades.copy()
    t["exit_ts"] = t.ts + (t.held.astype(int) + 1) * BAR_MS
    events = []
    for i, row in t.iterrows():
        events.extend([(int(row.ts), 1, "in", i), (int(row.exit_ts), 0, "out", i)])
    events.sort(key=lambda x: (x[0], x[1]))
    days = pd.date_range(
        pd.to_datetime(t.ts.min(), unit="ms", utc=True).normalize(),
        pd.to_datetime(t.exit_ts.max(), unit="ms", utc=True).normalize(),
        freq="D", tz="UTC")
    equity, notional, values, cursor = 1.0, {}, [], 0
    for day in days:
        limit = int((day + pd.Timedelta(days=1)).timestamp() * 1000)
        while cursor < len(events) and events[cursor][0] < limit:
            _, _, kind, idx = events[cursor]
            row = t.iloc[idx]
            if kind == "out":
                equity += notional.pop(idx) * float(row.net)
            else:
                notional[idx] = equity * weight
            cursor += 1
        values.append(equity)
    return pd.Series(values, index=days)


def summarize(trades: pd.DataFrame, equity: pd.Series, skipped: int, weight: float) -> dict:
    if trades.empty or equity.empty:
        return {"n_trades": 0, "skipped_entries": skipped}
    returns = equity.pct_change().dropna()
    span_days = max((equity.index[-1] - equity.index[0]).days, 1)
    years = span_days / 365.25
    total = float(equity.iloc[-1] - 1.0)
    hold = trades.held.astype(float) * 0.25
    recovery = _recovery_stats(equity)
    return {
        "return_basis": "sample_interval_cumulative",
        "sample_days": float(span_days),
        "total_return": total,
        "annualized_return": (1 + total) ** (1 / years) - 1 if total > -1 else -1.0,
        "max_drawdown_pct": float((equity / equity.cummax() - 1).min()),
        "sharpe_daily": float(returns.mean() / returns.std(ddof=1) * np.sqrt(365)) if returns.std(ddof=1) else np.nan,
        "vol_annualized": float(returns.std(ddof=1) * np.sqrt(365)),
        "n_trades": int(len(trades)),
        "win_rate": float((trades.net > 0).mean()),
        "avg_net_bps": float(trades.net.mean() * 1e4),
        "avg_position_pct": float((hold * weight).sum() / (span_days * 24)),
        "avg_hold_hours": float(hold.mean()),
        "median_hold_hours": float(hold.median()),
        "avg_concurrent": float(hold.sum() / (span_days * 24)),
        "skipped_entries": int(skipped),
        **recovery,
    }


def run(oos_path: Path, output: Path, cost_bps: float = 10.0) -> pd.DataFrame:
    oos = pd.read_csv(oos_path)
    candidates = bd.simulate(
        oos, "c", 0.01, "both", cost_bps, "model", np.random.default_rng(11),
        separate_sides=False,
    )
    ref = oos[["ts", "symbol", "p_up", "hi_0.01", "lo_0.01", "width_c"]]
    candidates = candidates.merge(ref, on=["ts", "symbol"], how="left")
    candidates["strength"] = _strength(candidates)
    rows = []
    for mode in ("3", "4", "5", "8", "dynamic"):
        selected, skipped = select_slots(candidates, mode)
        # Keep the original 1/8 risk budget; slot count is capacity, not leverage.
        metrics = summarize(selected, shared_curve(selected, 1 / 8), skipped, 1 / 8)
        metrics.update(strategy=f"slots_{mode}", cost_bps=cost_bps, position_weight=0.125)
        rows.append(metrics)
    result = pd.DataFrame(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(result.to_string(index=False))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--oos", type=Path, default=RESULTS / "oos_dir_c_roll730.csv.gz")
    parser.add_argument("--output", type=Path, default=RESULTS / "slot_sweep_c_roll730.csv")
    parser.add_argument("--cost-bps", type=float, default=10.0)
    args = parser.parse_args()
    run(args.oos, args.output, args.cost_bps)
