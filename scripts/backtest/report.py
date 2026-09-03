"""Portfolio numbers for a chosen set of cells on the extended 2021-2026 sample.

Reports each cell at both cost assumptions, and splits the OOS window into the
part that is genuinely new (2023-04 -> 2024-11, never seen by the 3-year study)
and the part that overlaps the earlier study's OOS.  The split matters: the
extended sample was built AFTER seeing the 3-year results, so only the new
segment is an untouched test of the earlier conclusion.

Usage: python3 -m scripts.backtest.report
"""
from __future__ import annotations

import json
import hashlib
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path

from scripts.data import build
from scripts.backtest import backtest_dir as bd
from scripts.backtest import portfolio
from scripts.backtest.backtest import summarise

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "crypto"
SPLIT = pd.Timestamp("2024-11-01", tz="UTC")   # start of the 3-year study's OOS
# The operational comparison set.  Keep the lower-level backtest capable of
# scanning every tail, but make the default report answer the live decision.
CELLS = [
    ("b", 0.01, "both"), ("b", 0.02, "both"),
    ("c", 0.01, "both"), ("c", 0.02, "both"),
]


def file_fingerprint(paths: list[Path]) -> str:
    """Short content fingerprint for the inputs that define this run."""
    h = hashlib.sha256()
    for path in sorted(paths):
        h.update(str(path.relative_to(ROOT)).encode())
        h.update(path.read_bytes())
    return h.hexdigest()[:12]


strategy_files = [Path(__file__), ROOT / "scripts" / "backtest" / "backtest_dir.py",
                  ROOT / "scripts" / "backtest" / "portfolio.py",
                  ROOT / "scripts" / "data" / "build.py",
                  ROOT / "scripts" / "modeling" / "train.py",
                  ROOT / "scripts" / "modeling" / "train_dir.py"]
strategy_fingerprint = file_fingerprint(strategy_files)
input_files = [RESULTS / f"oos_dir_{cfg}.csv.gz"
               for cfg in sorted({c for c, _, _ in CELLS})]
data_fingerprint = file_fingerprint(input_files)
run_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
run_id = f"{run_stamp}-{strategy_fingerprint}-{data_fingerprint}"
RUN_DIR = RESULTS / "runs" / run_id
suffix = 1
while RUN_DIR.exists():
    run_id = f"{run_stamp}-{strategy_fingerprint}-{data_fingerprint}-{suffix}"
    RUN_DIR = RESULTS / "runs" / run_id
    suffix += 1
RUN_DIR.mkdir(parents=True, exist_ok=False)

oos = {}
for cfg in sorted({c for c, _, _ in CELLS}):
    o = pd.read_csv(RESULTS / f"oos_dir_{cfg}.csv.gz")
    o["dt"] = pd.to_datetime(o["dt"], utc=True)
    oos[cfg] = o
yrs = {c: (o.dt.max() - o.dt.min()).total_seconds() / (365.25 * 86400)
       for c, o in oos.items()}

# Keep the provenance next to the numbers.  These are deliberately explicit:
# a result copied out of results/ must still say what was tested.
meta = {
    "run_id": run_id,
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "strategy_fingerprint": strategy_fingerprint,
    "data_fingerprint": data_fingerprint,
    "strategy_files": [str(p.relative_to(ROOT)) for p in strategy_files],
    "input_oos_files": [str(p.relative_to(ROOT)) for p in input_files],
    "data_source": "Binance data.binance.vision USDT-M perpetual archives",
    "feature_source": "Binance 15m klines + 5m derivatives metrics folded to 15m",
    "label_source": "Binance USDT-M 15m OHLC, triple-barrier labels",
    "execution_venue_assumption": "OKX USDT perpetual (cost stress tested at 10 and 16 bps)",
    "timeframe": "15m",
    "symbols": list(bd.SYMBOLS) if hasattr(bd, "SYMBOLS") else ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
    "sample_start_utc": min(o.dt.min() for o in oos.values()).isoformat(),
    "sample_end_utc": max(o.dt.max() for o in oos.values()).isoformat(),
    "oos_start_utc": min(o.dt.min() for o in oos.values()).isoformat(),
    "oos_end_utc": max(o.dt.max() for o in oos.values()).isoformat(),
    "model": "LightGBM direction-pure expanding walk-forward, 6 folds, purged embargo",
    "configs": {name: {"barrier_k": k, "horizon_bars": h}
                for name, k, h in build.CONFIGS if name in {c for c, _, _ in CELLS}},
    "cells": [{"config": c, "tail": tl, "policy": pol} for c, tl, pol in CELLS],
    "policies": ["both"],
    "variant": "model",
    "cost_bps": [10.0, 16.0],
    "selection_note": "Four operational cells; a and other tails remain available for research but are excluded here.",
}
print("=== 实验信息 ===")
for key in ("data_source", "feature_source", "timeframe", "symbols", "oos_start_utc", "oos_end_utc", "model"):
    print(f"{key}: {meta[key]}")

hdr = (f"{'cell':>16} {'cost':>5} {'n':>5} {'/mo':>5} {'win%':>6} {'net':>7} {'t':>6} "
       f"{'年化':>7} {'回撤':>7} {'夏普':>6} {'波动':>6} {'在场':>6}")
print(hdr); print("-" * len(hdr))
rows = []
for cfg, tl, pol in CELLS:
    tr = bd.simulate(oos[cfg], cfg, tl, pol, 10.0, "model", np.random.default_rng(11))
    if tr.empty:
        continue
    for cost in (10.0, 16.0):
        t2 = tr.copy()
        t2["net"] = t2["gross"] - cost / 1e4
        s = summarise(t2, yrs[cfg]); p = portfolio.stats(t2)
        if not p:
            continue
        print(f"{cfg+' t='+str(tl)+' '+pol:>16} {cost:>5.0f} {s['trades']:>5.0f} "
              f"{s['per_month']:>5.1f} {s['winrate']:>5.1%} {s['net_bps']:>+7.1f} "
              f"{s['t_stat']:>+6.2f} {p['cagr']:>+6.1%} {p['max_dd']:>+6.1%} "
              f"{p['sharpe']:>6.2f} {p['vol_ann']:>5.1%} {p['days_in_market']:>5.1%}")
        rows.append({"cfg": cfg, "tail": tl, "policy": pol, "cost": cost,
                     **s, **{f"pf_{k}": v for k, v in p.items()}})
report = pd.DataFrame(rows)
for key, value in meta.items():
    if key not in report.columns:
        report[key] = json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
report.to_csv(RUN_DIR / "report_long.csv", index=False)

print(f"\n=== 分段：2023-04→2024-11 是本次扩样才出现的、之前研究从未见过的样本外 ===")
hdr2 = (f"{'cell':>16} | {'新段 n':>7} {'win%':>6} {'net':>7} {'t':>6} "
        f"| {'旧段 n':>7} {'win%':>6} {'net':>7} {'t':>6}")
print(hdr2); print("-" * len(hdr2))
for cfg, tl, pol in CELLS:
    tr = bd.simulate(oos[cfg], cfg, tl, pol, 10.0, "model", np.random.default_rng(11))
    if tr.empty:
        continue
    new, old = tr[tr.dt < SPLIT], tr[tr.dt >= SPLIT]
    if len(new) < 20 or len(old) < 20:
        continue
    a = summarise(new, 1.0); b = summarise(old, 1.0)
    print(f"{cfg+' t='+str(tl)+' '+pol:>16} | {a['trades']:>7.0f} {a['winrate']:>5.1%} "
          f"{a['net_bps']:>+7.1f} {a['t_stat']:>+6.2f} | {b['trades']:>7.0f} "
          f"{b['winrate']:>5.1%} {b['net_bps']:>+7.1f} {b['t_stat']:>+6.2f}")

# A single aggregate can hide a regime or a symbol doing all the work.
# Export the same 10bps series split by year and symbol for a quick audit.
stability = []
for cfg, tl, pol in CELLS:
    tr = bd.simulate(oos[cfg], cfg, tl, pol, 10.0, "model", np.random.default_rng(11))
    if tr.empty:
        continue
    tr["year"] = tr["dt"].dt.year
    for dimension, groups in (("year", tr.groupby("year")),
                              ("symbol", tr.groupby("symbol"))):
        for value, group in groups:
            s = summarise(group, 1.0)
            stability.append({"cfg": cfg, "tail": tl, "policy": pol,
                              "cost_bps": 10.0, "breakdown": dimension,
                              "segment": str(value), **s})
pd.DataFrame(stability).to_csv(RUN_DIR / "report_stability.csv", index=False)
with (RUN_DIR / "report_meta.json").open("w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

# Keep stable convenience paths for scripts and humans.  The immutable run
# directory above is the audit record; these files are explicitly "latest".
latest = RESULTS / "latest"
latest.mkdir(exist_ok=True)
for name in ("report_long.csv", "report_stability.csv", "report_meta.json"):
    target = latest / name
    target.write_bytes((RUN_DIR / name).read_bytes())
    (RESULTS / name).write_bytes((RUN_DIR / name).read_bytes())
print(f"\nrun_id: {run_id}")
print(f"已写入 results/crypto/runs/{run_id}/，并更新 results/crypto/latest/")
