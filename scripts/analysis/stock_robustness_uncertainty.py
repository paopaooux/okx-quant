"""Paired seven-day block resampling and contribution concentration diagnostics."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analysis.stock_funding_robustness import OUT as FUNDED
from scripts.analysis.stock_robustness_study import OUT as STUDY, VARIANTS

OUT = Path("results/stock_robustness_uncertainty_20260928")
SEED = 20260928
REPETITIONS = 10000


def block_indices(days, repetitions=REPETITIONS, block=7, seed=SEED):
    if days < block:
        raise ValueError("Sample shorter than a resampling block")
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, days-block+1, size=(repetitions, int(np.ceil(days/block))))
    return (starts[:, :, None]+np.arange(block)).reshape(repetitions, -1)[:, :days]


def bootstrap_pair(curve, baseline, repetitions=REPETITIONS, comparisons=25):
    if not curve.index.equals(baseline.index) or curve.isna().any() or baseline.isna().any():
        raise ValueError("Paired daily curves must have identical complete clocks")
    if (curve <= 0).any() or (baseline <= 0).any():
        raise ValueError("Log returns require positive equity")
    a = np.diff(np.log(np.r_[1., curve.to_numpy()]))
    b = np.diff(np.log(np.r_[1., baseline.to_numpy()]))
    indices = block_indices(len(a), repetitions)
    ra = np.expm1(a[indices].sum(axis=1))*100
    rb = np.expm1(b[indices].sum(axis=1))*100
    delta = ra-rb
    family_tail = .05/(2*comparisons)
    return dict(days=len(a), return_pct=(curve.iloc[-1]-1)*100,
                paired_delta_pp=(curve.iloc[-1]-baseline.iloc[-1])*100,
                return_ci_low=np.quantile(ra, .025), return_ci_high=np.quantile(ra, .975),
                delta_ci_low=np.quantile(delta, .025), delta_ci_high=np.quantile(delta, .975),
                family_delta_ci_low=np.quantile(delta, family_tail),
                family_delta_ci_high=np.quantile(delta, 1-family_tail),
                bootstrap_fraction_delta_positive=float((delta > 0).mean()))


def concentration(trades):
    contribution = trades.notional*trades.net
    symbols = contribution.groupby(trades.symbol).sum().sort_values(ascending=False)
    days = contribution.groupby(pd.to_datetime(trades.entry_ts, utc=True).dt.normalize()).sum().sort_values(ascending=False)
    positive = symbols[symbols > 0]
    return dict(trades=len(trades), symbols=len(symbols),
                top_symbol=symbols.index[0], top_symbol_contribution_pp=symbols.iloc[0]*100,
                top_positive_symbol_share=positive.max()/positive.sum() if len(positive) else np.nan,
                top_three_positive_share=positive.head(3).sum()/positive.sum() if len(positive) else np.nan,
                best_entry_day_contribution_pp=days.iloc[0]*100,
                fixed_book_return_without_best_symbol_pct=(contribution.sum()-symbols.iloc[0])*100)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    hashes, rows, contributions = {}, [], []

    def read(path, curve=False):
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        frame = pd.read_csv(path)
        if not curve:
            return frame
        return pd.Series(frame.equity.to_numpy(), index=pd.to_datetime(frame.iloc[:, 0], utc=True))

    scenarios = [(STUDY, p, "unfunded") for p in ("historical", "development", "june_july", "august", "september", "recent")]
    scenarios += [(FUNDED, p, "funded_conservative") for p in ("july_september", "july", "august", "september", "recent")]
    for root, period, scenario in scenarios:
        suffix = "" if scenario == "unfunded" else "_funded_conservative"
        base = read(root / f"daily_baseline_{period}_68{suffix}.csv", curve=True)
        for variant in VARIANTS:
            tag = f"{variant.name}_{period}_68{suffix}"
            curve = read(root / f"daily_{tag}.csv", curve=True)
            tr = read(root / f"trades_{tag}.csv")
            rows.append(dict(variant=variant.name, period=period, scenario=scenario,
                             **bootstrap_pair(curve, base)))
            contributions.append(dict(variant=variant.name, period=period, scenario=scenario,
                                      **concentration(tr)))
    pd.DataFrame(rows).to_csv(OUT / "bootstrap.csv", index=False)
    pd.DataFrame(contributions).to_csv(OUT / "concentration.csv", index=False)
    capacity_rows, capacity_months, capacity_concentration = [], [], []
    portfolio = Path("results/stock_portfolio_robustness_20260928")
    for scope in ("stock_only", "crypto_only", "shared"):
        for sizing in ("fixed_20pct", "equal_slot_budget"):
            base = read(portfolio / f"daily_capacity_{scope}_5_{sizing}_68.csv", curve=True)
            for slots in range(1, 6):
                tag = f"capacity_{scope}_{slots}_{sizing}_68"
                curve = read(portfolio / f"daily_{tag}.csv", curve=True)
                tr = read(portfolio / f"trades_{tag}.csv")
                capacity_rows.append(dict(scope=scope, sizing=sizing, slots=slots,
                                          **bootstrap_pair(curve, base, comparisons=24)))
                capacity_concentration.append(dict(scope=scope, sizing=sizing, slots=slots,
                                                    **concentration(tr)))
                ends = curve.groupby(curve.index.strftime("%Y-%m")).last()
                changes = (ends/ends.shift().fillna(1)-1)*100
                capacity_months.extend(dict(scope=scope, sizing=sizing, slots=slots, month=m, return_pct=r)
                                       for m, r in changes.items())
    pd.DataFrame(capacity_rows).to_csv(OUT / "capacity_bootstrap.csv", index=False)
    pd.DataFrame(capacity_concentration).to_csv(OUT / "capacity_concentration.csv", index=False)
    pd.DataFrame(capacity_months).to_csv(OUT / "capacity_monthly.csv", index=False)
    (OUT / "manifest.json").write_text(json.dumps(dict(seed=SEED, repetitions=REPETITIONS,
        block_days=7, cost_bps=68, input_hashes=hashes,
        caveats=["Ordinary intervals are exploratory/unadjusted; family_delta bounds use Bonferroni tails for 25 comparisons.",
                 "Family bounds are approximate block-bootstrap percentiles with only about 10 simulated observations in each extreme tail.",
                 "Capacity diagnostics use a separate family of 24 non-five-slot comparisons across three scopes and two sizing conventions.",
                 "Resamples daily marked log returns in seven-consecutive-calendar-day blocks; partial boundary days retained.",
                 "Intervals describe this retrospective sample, not future-regime, model or execution uncertainty.",
                 "Bootstrap positive fraction is not a posterior probability or a calibrated hypothesis-test p-value.",
                 "Removing best-symbol contribution holds the admitted book fixed; it is an influence diagnostic, not a tradable exclusion rule.",
                 "Overlapping periods are not independent validation evidence."]), indent=2))
    print(pd.DataFrame(rows).query("variant in ['baseline', 'displacement_le_8pct', 'stock_cap_2'] and period in ['historical', 'july_september']").to_string(index=False))


if __name__ == "__main__":
    main()
