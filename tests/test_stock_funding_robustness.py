from types import SimpleNamespace

import numpy as np
import pandas as pd

from scripts.analysis.stock_funding_robustness import funding_return
from scripts.analysis.stock_robustness_study import STEP, marked_curve


def fixture(side):
    index = pd.date_range("2026-07-01T00:00Z", periods=20, freq=STEP)
    frame = pd.DataFrame({"close": 100.}, index=index)
    rates = pd.Series([.01, .01, .02, .01], index=pd.DatetimeIndex([index[0], index[3], index[6], index[15]]))
    row = SimpleNamespace(entry_ts=index[2], exit_ts=index[6], coverage_deadline=index[12],
                          exit_lower_ts=index[5], side=side, entry_price=100.)
    return row, rates, frame


def test_long_pays_positive_rate_and_boundary_debit_is_included():
    row, rates, frame = fixture(1)
    result = funding_return(row, rates, frame)
    assert np.isclose(result["funding"], -.03)
    assert np.isclose(result["funding_optimistic"], -.01)
    assert result["ambiguous_settlements"] == 1


def test_short_receives_but_boundary_credit_is_excluded():
    row, rates, frame = fixture(-1)
    result = funding_return(row, rates, frame)
    assert np.isclose(result["funding"], .01)
    assert np.isclose(result["funding_optimistic"], .03)


def test_coverage_is_not_assumed_zero_and_mark_uses_completed_bar():
    row, rates, frame = fixture(1)
    assert funding_return(row, rates.iloc[1:], frame) is None
    row.exit_ts = rates.index[1]
    row.exit_lower_ts = row.exit_ts
    before = funding_return(row, rates, frame)
    frame.loc[row.exit_ts:, "close"] = 10000.
    after = funding_return(row, rates, frame)
    assert before == after


def test_funding_enters_marked_equity_at_settlement_without_double_count():
    row, _, frame = fixture(1)
    record = dict(symbol="TEST", entry_ts=row.entry_ts, exit_ts=row.exit_ts,
                  entry_price=100., side=1, notional=.2, cost_bps=0., net=-.01,
                  funding_events=[(str(row.entry_ts+STEP), -.01)])
    eq = marked_curve(pd.DataFrame([record]), {"TEST": frame}, frame.index[0], frame.index[15])
    assert eq.loc[row.entry_ts] == 1.
    assert np.isclose(eq.loc[row.entry_ts+STEP], .998)
    assert np.isclose(eq.iloc[-1], .998)
