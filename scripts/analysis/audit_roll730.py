"""Audit the frozen rolling-730-day crypto candidate.

This report intentionally does not search parameters.  It evaluates the fixed
strategy card and writes a machine-readable audit containing the temporal
holdout, cost stress, yearly stability, and null controls.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.backtest import backtest_dir as bd
from scripts.backtest import portfolio


def _clean(value):
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _stats(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {"trades": 0}
    out = portfolio.stats(trades)
    out["trades"] = int(len(trades))
    out["avg_net_bps"] = float(trades["net"].mean() * 10_000)
    return _clean(out)


def audit(oos_path: Path, tail: float, split: str) -> dict:
    oos = pd.read_csv(oos_path)
    oos["dt"] = pd.to_datetime(oos["dt"], utc=True)
    split_ts = pd.Timestamp(split, tz="UTC")
    rng = np.random.default_rng(11)
    trades = bd.simulate(oos, "c", tail, "both", 10.0, "model", rng)
    trades["dt"] = pd.to_datetime(trades["dt"], utc=True)
    selection = trades[trades["dt"] < split_ts]
    validation = trades[trades["dt"] >= split_ts]

    costs = {}
    for cost in (6.0, 8.0, 10.0, 12.0, 16.0, 20.0):
        t = bd.simulate(oos, "c", tail, "both", cost, "model", np.random.default_rng(11))
        costs[str(int(cost))] = _stats(t)

    yearly = {}
    for year, group in trades.groupby(trades["dt"].dt.year):
        yearly[str(year)] = _stats(group)

    controls = {}
    for variant in ("inverted", "random"):
        t = bd.simulate(oos, "c", tail, "both", 10.0, variant, np.random.default_rng(11))
        controls[variant] = _stats(t)

    return _clean({
        "strategy": "crypto_direction_c_rolling_730d_tail_0.01",
        "oos_file": str(oos_path),
        "symbols": sorted(oos["symbol"].astype(str).unique()),
        "sample_start_utc": oos["dt"].min().isoformat(),
        "sample_end_utc": oos["dt"].max().isoformat(),
        "selection_split_utc": split_ts.isoformat(),
        "tail": tail,
        "selection_pre_split": _stats(selection),
        "frozen_validation": _stats(validation),
        "full_oos_10bp": _stats(trades),
        "cost_stress": costs,
        "yearly_10bp": yearly,
        "null_controls_10bp": controls,
    })


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oos", type=Path, default=Path("results/crypto/oos_dir_c_roll730.csv.gz"))
    parser.add_argument("--tail", type=float, default=0.01)
    parser.add_argument("--split", default="2025-01-01")
    parser.add_argument("--output", type=Path, default=Path("results/crypto/roll730_report.json"))
    args = parser.parse_args()
    result = audit(args.oos, args.tail, args.split)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
