"""Verify frozen provenance and all published research table/trade identities."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analysis.stock_robustness_study import OUT as STUDY

ROOTS = {
    "main": STUDY,
    "funding": Path("results/stock_funding_robustness_20260928"),
    "execution": Path("results/stock_execution_audit_20260928"),
    "portfolio": Path("results/stock_portfolio_robustness_20260928"),
    "uncertainty": Path("results/stock_robustness_uncertainty_20260928"),
    "confirmation": Path("results/stock_robustness_confirmation_20260928"),
    "capacity_confirmation": Path("results/stock_capacity_confirmation_20260928"),
}


def verify_trade_summary(row, path):
    tr = pd.read_csv(path)
    assert len(tr) == row.trades, path
    assert np.isclose((tr.notional*tr.net).sum()*100, row.return_pct, atol=1e-8), path
    assert np.isclose(tr.net.gt(0).mean()*100, row.win_pct, atol=1e-8), path
    assert np.isfinite(tr[["notional", "net", "entry_price"]]).all().all(), path
    assert tr.notional.gt(0).all() and tr.entry_price.gt(0).all(), path
    assert not tr.duplicated(["symbol", "entry_ts"]).any(), path
    entry, exit_ = pd.to_datetime(tr.entry_ts, utc=True), pd.to_datetime(tr.exit_ts, utc=True)
    assert exit_.ge(entry).all(), path
    assert entry.is_monotonic_increasing, path
    if "coverage_deadline" in tr:
        assert exit_.le(pd.to_datetime(tr.coverage_deadline, utc=True)).all(), path
    if "cost_bps" in tr:
        funding = tr.funding.fillna(0.) if "funding" in tr else 0.
        np.testing.assert_allclose(tr.net, tr.gross+funding-tr.cost_bps/10000, atol=1e-12)


def main():
    digests, manifests, checked = {}, {}, 0

    def digest(path):
        key = str(path)
        if key not in digests:
            digests[key] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        return digests[key]

    for name, root in ROOTS.items():
        manifests[name] = json.loads((root / "manifest.json").read_text())
        for path, expected in manifests[name].get("input_hashes", {}).items():
            assert digest(path) == expected, f"Changed frozen input: {path}"
    assert digest("docs/stock_robustness_protocol_20260928.md") == manifests["main"]["protocol_sha256"]
    assert digest("docs/stock_robustness_followup_20260928.md") == manifests["confirmation"]["followup_protocol_sha256"]
    assert digest(Path(manifests["funding"]["rate_source"]) / "manifest.json") == manifests["funding"]["rate_manifest_sha256"]
    assert digest("docs/stock_capacity_followup_20260928.md") == manifests["capacity_confirmation"]["protocol_sha256"]
    for name, expected in (("main", 312), ("funding", 520), ("execution", 18),
                           ("portfolio", 114), ("confirmation", 126), ("capacity_confirmation", 60)):
        root = ROOTS[name]
        summary = pd.read_csv(root / "summary.csv")
        assert len(summary) == expected, (name, len(summary), expected)
        for row in summary.itertuples():
            if name == "main":
                tag = f"{row.variant}_{row.period}_{row.cost}"
            elif name == "funding":
                tag = f"{row.variant}_{row.period}_{row.cost}_{row.scenario}"
            elif name == "execution":
                if row.stage == "archived_reported":
                    continue
                tag = f"{row.stage}_{row.period}_{row.cost}"
            elif name == "portfolio":
                tag = f"{row.variant}_{row.scenario}_{row.period}_{row.cost}_{row.scope}"
            elif name == "confirmation":
                tag = f"{row.variant}_{row.delay_minutes}_{row.scenario}_{row.period}_{row.cost}"
            else:
                tag = f"{row.scenario}_{row.delay_minutes}_{row.slots}_{row.allocation}"
            verify_trade_summary(row, root / f"trades_{tag}.csv")
            checked += 1
    capacity = pd.read_csv(ROOTS["portfolio"] / "capacity_summary.csv")
    assert len(capacity) == 60
    assert capacity.max_positions.le(capacity.slots).all()
    assert capacity.longest_drawdown_days.ge(0).all()
    for row in capacity.itertuples():
        path = ROOTS["portfolio"] / f"trades_capacity_{row.scope}_{row.slots}_{row.sizing}_{row.cost}.csv"
        verify_trade_summary(row, path)
        checked += 1
    folds = pd.read_csv(STUDY / "fold_results.csv")
    assert len(folds) == 8 and len(pd.read_csv(STUDY / "fold_training.csv")) == 104
    for label, group in folds.groupby("label"):
        assert np.isclose((1+group.return_pct/100).prod(), manifests["main"]["fold_growth"][label])
    for table, count in (("bootstrap", 286), ("concentration", 286), ("capacity_bootstrap", 30),
                         ("capacity_monthly", 210), ("capacity_concentration", 30)):
        assert len(pd.read_csv(ROOTS["uncertainty"] / f"{table}.csv")) == count
    coverage = pd.read_csv(STUDY / "coverage.csv")
    assert len(coverage) == 160 and coverage.rows.sum() == 5069687
    assert coverage.missing_bars.sum() == 1
    funding = pd.read_csv(ROOTS["funding"] / "coverage.csv")
    assert len(funding) == 26 and funding.excluded.eq(0).all() and funding.covered.eq(849).all()
    assert manifests["main"]["candidate_bridge_reproduced"] == manifests["execution"]["reproduced_candidates"] == 1049
    for path in Path("scripts/analysis").glob("*stock*robustness*.py"):
        digest(path)
    for path in ("scripts/analysis/stock_execution_audit.py", "scripts/analysis/download_stock_funding.py",
                 "scripts/analysis/stock_capacity_confirmation.py",
                 "scripts/analysis/strategy_replacement_study.py"):
        digest(path)
    for root in ROOTS.values():
        for path in root.glob("*summary.csv"):
            digest(path)
    result = dict(verified=True, trade_summary_pairs=checked, hashed_files=len(digests),
                  hashes=digests, caveat="Integrity and accounting verification, not proof of strategy profitability or future execution.")
    (STUDY / "verification.json").write_text(json.dumps(result, indent=2))
    print(f"Verified {checked} trade/summary pairs and {len(digests)} source/code/result hashes.")


if __name__ == "__main__":
    main()
