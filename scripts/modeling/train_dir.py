"""Direction-pure model: among bars that resolve, which barrier fires first?

train.py's label conflates two questions.  "Does the upper barrier fire within H
bars" is high when the market is *about to move at all*, regardless of which
way, so a model can raise its precision from 20% to 47% purely by timing
volatility -- and the diagnostic confirmed exactly that (at q=0.999 the long
signal showed P(long win)=46.7% against P(short win)=44.0%, a 90.7% touch rate
with a coin-flip inside it).  Precision on that label is not a win rate.

Closing the channel: fit only on bars where some barrier fired, with the target
"upper fired".  Touch rate is now constant by construction at 100%, the base
rate sits near 50%, and the only way to beat it is to know the direction.

Conditioning on resolution uses information unavailable at entry, which is fine
for *fitting* -- it just focuses the model on the directional question -- but
never for scoring.  Scoring runs on every bar with the realised P&L, so nothing
downstream inherits the conditioning.

Usage:  python3 -m scripts.modeling.train_dir [config] [--no-session]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

from scripts.data import build
from scripts.modeling.train import PARAMS, feature_columns, folds, BAR_MS, TRAIN_FRAC, N_FOLDS

ROOT = Path(__file__).resolve().parents[2]
DATA, RESULTS, MODELS = ROOT / "data", ROOT / "results" / "crypto", ROOT / "models"
# Cuts are two-tailed: the high tail is a long, the low tail is a short.
TAILS = (0.10, 0.05, 0.02, 0.01, 0.005, 0.001)


def run(config: str, drop_session: bool, price_source: str = "binance",
        roll_days: int | None = None, panel_path: str | Path | None = None,
        dynamic_universe: bool = False, dynamic_prior_n: int = 50,
        dynamic_min_signals: int = 20, dynamic_max_symbols: int = 6,
        symbols: tuple[str, ...] | None = None) -> None:
    """roll_days: train on a trailing window of that many days instead of an
    expanding one.

    Motivation, from the expanding-window run: early stopping picked
    34 -> 129 -> 36 -> 9 -> 1 -> 3 trees across the six folds.  A tree count that
    collapses to one as the training set GROWS means the added history is
    actively unhelpful -- the relationship is non-stationary and old data drags
    the fit toward a regime that no longer holds.  A trailing window is the
    standard remedy and is the one modelling change with the right order of
    magnitude.
    """
    # "okx" swaps in labels and exits computed from OKX swap prices while the
    # features stay Binance-derived -- the venue-portability test.
    src = "panel_okx.csv.gz" if price_source == "okx" else "panel.csv.gz"
    panel_file = Path(panel_path) if panel_path else DATA / src
    if not panel_file.is_absolute():
        panel_file = ROOT / panel_file
    panel = pd.read_csv(panel_file)
    if symbols:
        panel = panel.loc[panel["symbol"].astype(str).isin(symbols)].copy()
    panel["dt"] = pd.to_datetime(panel["dt"], utc=True)
    panel = panel.sort_values(["ts", "symbol"]).reset_index(drop=True)

    H = dict((n, h) for n, _, h in build.CONFIGS)[config]
    embargo = H * BAR_MS
    feats = feature_columns(panel)
    if drop_session:
        feats = [c for c in feats if c not in ("hour", "dow")]
    panel["sym_code"] = panel["symbol"].astype("category").cat.codes
    feats = feats + ["sym_code"]

    lw = panel[f"long_win_{config}"].to_numpy(np.int8)
    sw = panel[f"short_win_{config}"].to_numpy(np.int8)
    resolved = (lw + sw) == 1
    y = lw
    tag = (f"{config}{'_nosession' if drop_session else ''}"
           f"{'_okx' if price_source == 'okx' else ''}"
           f"{f'_roll{roll_days}' if roll_days else ''}"
           f"{f'_u{len(symbols)}' if symbols else ''}"
           f"{f'_dyn_prior{dynamic_prior_n}' if dynamic_universe else ''}")
    print(f"direction model  config {config}  H={H}  features={len(feats)}"
          f"{'  (hour/dow dropped)' if drop_session else ''}"
          f"{'  [OKX prices]' if price_source == 'okx' else ''}")
    print(f"  resolved bars {resolved.sum():,}/{len(panel):,} ({resolved.mean():.1%})"
          f"   base P(up first | resolved) = {y[resolved].mean():.1%}\n")

    times = np.sort(panel["ts"].unique())
    ts = panel["ts"].to_numpy()
    X = panel[feats]

    oos = []
    dynamic_rows = []
    for fi, (a, b) in enumerate(folds(times)):
        t_lo, t_hi = times[a], times[b - 1]
        test = (ts >= t_lo) & (ts <= t_hi)
        train = ts < (t_lo - embargo)
        if roll_days is not None:
            train &= ts >= (t_lo - embargo - roll_days * 86_400_000)
        tr_times = np.sort(np.unique(ts[train]))
        v_lo = tr_times[int(0.88 * len(tr_times))]
        valid = train & (ts >= v_lo) & resolved
        fit = train & (ts < (v_lo - embargo)) & resolved

        m = lgb.LGBMClassifier(**PARAMS)
        m.fit(X[fit], y[fit], eval_set=[(X[valid], y[valid])], eval_metric="auc",
              callbacks=[lgb.early_stopping(100, verbose=False)])
        MODELS.mkdir(exist_ok=True)
        m.booster_.save_model(str(MODELS / f"dir_{tag}_fold{fi}.txt"))

        rec = panel.loc[test, ["ts", "dt", "symbol", f"held_{config}",
                               f"exit_ret_{config}", f"long_win_{config}",
                               f"short_win_{config}", f"width_{config}"]].copy()
        rec["fold"] = fi
        rec["p_up"] = m.predict_proba(X[test])[:, 1]
        p_tr = m.predict_proba(X[fit])[:, 1]
        for t in TAILS:
            rec[f"hi_{t}"] = float(np.quantile(p_tr, 1 - t))
            rec[f"lo_{t}"] = float(np.quantile(p_tr, t))
        if dynamic_universe:
            # Universe selection is made only from the validation tail that is
            # not used to fit the trees.  It is therefore available before the
            # following test block and cannot inspect test outcomes.
            p_valid = m.predict_proba(X[valid])[:, 1]
            hi, lo = float(np.quantile(p_tr, 0.99)), float(np.quantile(p_tr, 0.01))
            valid_rows = panel.loc[valid, ["symbol", f"exit_ret_{config}"]].copy()
            valid_rows["p_up"] = p_valid
            valid_rows["side"] = np.where(valid_rows.p_up >= hi, 1,
                                           np.where(valid_rows.p_up <= lo, -1, 0))
            valid_rows = valid_rows.loc[valid_rows.side != 0]
            valid_rows["net"] = valid_rows.side * valid_rows[f"exit_ret_{config}"] - 10.0 / 1e4
            quality_all = valid_rows.groupby("symbol")["net"].agg(n="count", mean_net="mean")
            quality_all["score"] = quality_all["n"] / (quality_all["n"] + max(0, dynamic_prior_n)) * quality_all["mean_net"]
            quality = (quality_all.query("n >= @dynamic_min_signals and score > 0")
                       .sort_values(["score", "n"], ascending=False))
            if len(quality) >= 3:
                eligible = list(quality.head(dynamic_max_symbols).index.astype(str))
            else:
                fallback = (quality_all.query("n >= @dynamic_min_signals")
                             .sort_values(["score", "n"], ascending=False))
                eligible = list(fallback.head(3).index.astype(str)) or sorted(panel.loc[test, "symbol"].astype(str).unique())
            rec["eligible"] = rec["symbol"].astype(str).isin(eligible)
            for symbol in sorted(panel.loc[test, "symbol"].astype(str).unique()):
                row = quality_all.loc[symbol] if symbol in quality_all.index else None
                dynamic_rows.append({
                    "fold": fi,
                    "test_start": pd.to_datetime(t_lo, unit="ms", utc=True).isoformat(),
                    "symbol": symbol,
                    "valid_signals": int(row["n"]) if row is not None else 0,
                    "valid_mean_net_bps": float(row["mean_net"] * 10_000) if row is not None else float("nan"),
                    "valid_score_bps": float(row["score"] * 10_000) if row is not None else float("nan"),
                    "valid_min_signals": dynamic_min_signals,
                    "prior_n": dynamic_prior_n,
                    "eligible": symbol in eligible,
                    "eligible_symbols": ",".join(eligible),
                })
        oos.append(rec)
        auc = m.best_score_["valid_0"]["auc"]
        print(f"  fold {fi}: fit {int(fit.sum()):,}  test {int(test.sum()):,}  "
              f"trees {m.best_iteration_:>4}  valid AUC {auc:.4f}", flush=True)

    oos = pd.concat(oos, ignore_index=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    oos.to_csv(RESULTS / f"oos_dir_{tag}.csv.gz", index=False, compression="gzip")
    if dynamic_universe:
        pd.DataFrame(dynamic_rows).to_csv(RESULTS / f"dynamic_universe_{tag}.csv", index=False)

    # The decomposition that killed version one, repeated here.  A directional
    # model must move P(long win) and P(short win) in *opposite* directions; a
    # volatility model moves both up together.
    lwv = oos[f"long_win_{config}"].to_numpy()
    swv = oos[f"short_win_{config}"].to_numpy()
    p = oos["p_up"].to_numpy()
    print(f"\n{'signal':>14} {'n':>6} {'P(long win)':>12} {'P(short win)':>13} "
          f"{'touch':>7} {'dir edge':>9}")
    print(f"{'(all bars)':>14} {len(oos):>6} {lwv.mean():>11.1%} {swv.mean():>12.1%} "
          f"{lwv.mean()+swv.mean():>6.1%} {lwv.mean()-swv.mean():>+8.1%}")
    rows = []
    for t in TAILS:
        for name, sel, sign in (("long", p >= oos[f"hi_{t}"].mean(), +1),
                                ("short", p <= oos[f"lo_{t}"].mean(), -1)):
            n = int(sel.sum())
            if n < 30:
                continue
            pl, ps = lwv[sel].mean(), swv[sel].mean()
            edge = sign * (pl - ps)
            rows.append(dict(tail=t, side=name, n=n, p_long=pl, p_short=ps,
                             touch=pl + ps, dir_edge=edge))
            print(f"{name+' t='+str(t):>14} {n:>6} {pl:>11.1%} {ps:>12.1%} "
                  f"{pl+ps:>6.1%} {edge:>+8.1%}")
    pd.DataFrame(rows).to_csv(RESULTS / f"decomp_dir_{tag}.csv", index=False)

    gain = pd.Series(0.0, index=feats)
    for f in MODELS.glob(f"dir_{tag}_fold*.txt"):
        bst = lgb.Booster(model_file=str(f))
        gain += pd.Series(bst.feature_importance("gain"), index=bst.feature_name())
    top = (gain.sort_values(ascending=False).head(15) / gain.sum())
    print("\ntop-15 features by gain")
    for k, v in top.items():
        print(f"  {v:6.2%}  {k}")
    top.to_csv(RESULTS / f"importance_dir_{tag}.csv")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    roll = next((int(a.split("=")[1]) for a in sys.argv if a.startswith("--roll=")), None)
    panel = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--panel=")), None)
    prior_n = next((int(a.split("=", 1)[1]) for a in sys.argv if a.startswith("--dynamic-prior-n=")), 50)
    min_signals = next((int(a.split("=", 1)[1]) for a in sys.argv if a.startswith("--dynamic-min-signals=")), 20)
    max_symbols = next((int(a.split("=", 1)[1]) for a in sys.argv if a.startswith("--dynamic-max-symbols=")), 6)
    symbols_arg = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--symbols=")), None)
    symbols = tuple(s.strip() for s in symbols_arg.split(",") if s.strip()) if symbols_arg else None
    run(args[0] if args else "a", "--no-session" in sys.argv,
        "okx" if "--okx" in sys.argv else "binance", roll, panel,
        "--dynamic-universe" in sys.argv, prior_n, min_signals, max_symbols, symbols)
