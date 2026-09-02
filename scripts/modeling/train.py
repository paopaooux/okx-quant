"""Fit LightGBM barrier classifiers with purged, expanding walk-forward CV.

Three things make this different from a naive sklearn split, and all three
matter more than the model:

1.  *Expanding walk-forward.*  Every prediction is made by a model that saw only
    earlier data.  A shuffled K-fold on a price panel trains on the future and
    is worth nothing.

2.  *Purge + embargo.*  A triple-barrier label at bar t reads bars t+1..t+H, so
    the last H bars before a test block leak into it.  They are dropped from
    train.  Without this the model gets graded on samples it half-memorised.

3.  *Thresholds fixed on train, applied to test.*  "Take the top 1% of test-set
    scores" secretly uses the test distribution.  The cut comes from the train
    predictions' quantile instead, which is what a live system would have.

Controls run alongside every claim: `random` picks the same number of entries
uniformly, `inverted` takes the bottom of the score distribution.  If the model
carries information, random sits at the base rate and inverted sits below it.

Usage:  python3 -m scripts.modeling.train [config]     config in {a, b, c}, default a
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

from scripts.data import build

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"

N_FOLDS = 6
TRAIN_FRAC = 0.40           # first 40% of history is train-only, never scored
QUANTILES = (0.90, 0.95, 0.98, 0.99, 0.995, 0.999)
BAR_MS = 15 * 60 * 1000

PARAMS = dict(
    objective="binary",
    n_estimators=3000,
    learning_rate=0.02,
    num_leaves=31,
    min_child_samples=1000,   # overlapping labels inflate effective n; keep leaves fat
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.6,
    reg_lambda=10.0,
    max_bin=127,
    verbosity=-1,
    n_jobs=8,
)

DROP = {"ts", "dt", "entry_px", "symbol"}


def feature_columns(panel: pd.DataFrame) -> list[str]:
    out = []
    for c in panel.columns:
        if c in DROP:
            continue
        if c.startswith(("long_win_", "short_win_", "held_", "exit_ret_", "width_")):
            continue          # labels, outcomes, and width (a pure function of sigma)
        out.append(c)
    return out


def folds(times: np.ndarray) -> list[tuple[int, int]]:
    n = len(times)
    start = int(TRAIN_FRAC * n)
    edges = np.linspace(start, n, N_FOLDS + 1).astype(int)
    return [(edges[i], edges[i + 1]) for i in range(N_FOLDS)]


def precision_report(name: str, score: np.ndarray, y: np.ndarray,
                     cuts: dict[float, float], rng: np.random.Generator) -> pd.DataFrame:
    base = y.mean()
    rows = []
    for q in QUANTILES:
        for variant in ("model", "inverted", "random"):
            if variant == "model":
                sel = score >= cuts[q]
            elif variant == "inverted":
                sel = score <= cuts[round(1 - q, 6)]
            else:
                k = int((score >= cuts[q]).sum())
                sel = np.zeros(len(score), bool)
                if k:
                    sel[rng.choice(len(score), size=k, replace=False)] = True
            n = int(sel.sum())
            if n < 30:
                continue
            p = float(y[sel].mean())
            se = np.sqrt(base * (1 - base) / n)
            rows.append(dict(side=name, q=q, variant=variant, n=n, winrate=p,
                             base=base, lift=p - base, z=(p - base) / se if se else np.nan))
    return pd.DataFrame(rows)


def run(config: str) -> None:
    panel = pd.read_csv(DATA / "panel.csv.gz")
    panel["dt"] = pd.to_datetime(panel["dt"], utc=True)
    panel = panel.sort_values(["ts", "symbol"]).reset_index(drop=True)

    H = dict((n, h) for n, _, h in build.CONFIGS)[config]
    embargo_ms = H * BAR_MS
    feats = feature_columns(panel)
    panel["sym_code"] = panel["symbol"].astype("category").cat.codes
    feats = feats + ["sym_code"]
    print(f"config {config}  H={H} bars  features={len(feats)}  rows={len(panel):,}")

    times = np.sort(panel["ts"].unique())
    ts = panel["ts"].to_numpy()
    X = panel[feats]
    rng = np.random.default_rng(7)

    oos = []
    for fi, (a, b) in enumerate(folds(times)):
        t_lo, t_hi = times[a], times[b - 1]
        test = (ts >= t_lo) & (ts <= t_hi)
        train = ts < (t_lo - embargo_ms)
        # last 12% of the train window, itself purged, is the early-stopping set
        tr_times = np.sort(np.unique(ts[train]))
        v_lo = tr_times[int(0.88 * len(tr_times))]
        valid = train & (ts >= v_lo)
        fit = train & (ts < (v_lo - embargo_ms))

        rec = panel.loc[test, ["ts", "dt", "symbol", "entry_px",
                               f"width_{config}", f"held_{config}", f"exit_ret_{config}",
                               f"long_win_{config}", f"short_win_{config}"]].copy()
        rec["fold"] = fi

        for side in ("long", "short"):
            y = panel[f"{side}_win_{config}"].to_numpy(np.int8)
            m = lgb.LGBMClassifier(**PARAMS)
            m.fit(X[fit], y[fit],
                  eval_set=[(X[valid], y[valid])], eval_metric="auc",
                  callbacks=[lgb.early_stopping(100, verbose=False)])
            MODELS.mkdir(exist_ok=True)
            m.booster_.save_model(str(MODELS / f"{config}_{side}_fold{fi}.txt"))
            rec[f"p_{side}"] = m.predict_proba(X[test])[:, 1]
            # Threshold from the *train* score distribution -- see module docstring.
            p_tr = m.predict_proba(X[fit])[:, 1]
            for q in sorted(set(QUANTILES) | {round(1 - x, 6) for x in QUANTILES}):
                rec[f"cut_{side}_{q}"] = float(np.quantile(p_tr, q))
            rec[f"iters_{side}"] = m.best_iteration_

        oos.append(rec)
        print(f"  fold {fi}: fit {int(fit.sum()):,}  test {int(test.sum()):,}  "
              f"{pd.to_datetime(t_lo, unit='ms', utc=True):%Y-%m-%d} -> "
              f"{pd.to_datetime(t_hi, unit='ms', utc=True):%Y-%m-%d}  "
              f"trees L{rec['iters_long'].iloc[0]}/S{rec['iters_short'].iloc[0]}", flush=True)

    oos = pd.concat(oos, ignore_index=True)
    RESULTS.mkdir(exist_ok=True)
    oos.to_csv(RESULTS / f"oos_{config}.csv.gz", index=False, compression="gzip")

    reports = []
    for side in ("long", "short"):
        cuts = {q: oos[f"cut_{side}_{q}"].mean() for q in
                sorted(set(QUANTILES) | {round(1 - x, 6) for x in QUANTILES})}
        reports.append(precision_report(side, oos[f"p_{side}"].to_numpy(),
                                        oos[f"{side}_win_{config}"].to_numpy(), cuts, rng))
    rep = pd.concat(reports, ignore_index=True)
    rep.to_csv(RESULTS / f"precision_{config}.csv", index=False)

    print("\nout-of-sample precision (pooled over all folds)")
    print("  n is overlapping bars, not trades -- see backtest.py for the trade count")
    for side in ("long", "short"):
        print(f"\n  {side}   base rate {rep[rep.side==side]['base'].iloc[0]:.1%}")
        print(f"  {'q':>7} {'variant':>9} {'n':>7} {'winrate':>8} {'lift':>7} {'z':>7}")
        for _, r in rep[rep.side == side].iterrows():
            print(f"  {r['q']:>7.3f} {r['variant']:>9} {r['n']:>7.0f} "
                  f"{r['winrate']:>7.1%} {r['lift']:>+6.1%} {r['z']:>7.1f}")

    gain = pd.Series(0.0, index=feats)
    for f in MODELS.glob(f"{config}_long_fold*.txt"):
        bst = lgb.Booster(model_file=str(f))
        gain += pd.Series(bst.feature_importance("gain"), index=bst.feature_name())
    top = gain.sort_values(ascending=False).head(20) / gain.sum()
    print("\ntop-20 features by gain (long model, summed over folds)")
    for k, v in top.items():
        print(f"  {v:6.2%}  {k}")
    top.to_csv(RESULTS / f"importance_{config}.csv")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "a")
