import pandas as pd
import pytest

from scripts.analysis.verify_stock_robustness import verify_trade_summary


def test_trade_accounting_verifier_detects_inconsistent_costs(tmp_path):
    path = tmp_path / "trades.csv"
    tr = pd.DataFrame([dict(symbol="A", entry_ts="2026-07-01T00:00Z", exit_ts="2026-07-01T01:00Z",
                           coverage_deadline="2026-07-01T02:00Z", notional=.2, entry_price=100.,
                           gross=.02, funding=.001, cost_bps=68., net=.0142)])
    row = pd.Series(dict(trades=1, return_pct=.284, win_pct=100.))
    tr.to_csv(path, index=False)
    verify_trade_summary(row, path)
    tr["cost_bps"] = 44.
    tr.to_csv(path, index=False)
    with pytest.raises(AssertionError):
        verify_trade_summary(row, path)


def test_trade_accounting_verifier_rejects_duplicate_admission(tmp_path):
    path = tmp_path / "trades.csv"
    row = dict(symbol="A", entry_ts="2026-07-01T00:00Z", exit_ts="2026-07-01T01:00Z",
               notional=.2, entry_price=100., net=.01)
    pd.DataFrame([row, row]).to_csv(path, index=False)
    with pytest.raises(AssertionError):
        verify_trade_summary(pd.Series(dict(trades=2, return_pct=.4, win_pct=100.)), path)
