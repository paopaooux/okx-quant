from dataclasses import replace

import numpy as np
import pandas as pd

from scripts.analysis.stock_portfolio_robustness import (
    admit, drawdown_durations, portfolio_curve, prepare_crypto,
)
from scripts.analysis.stock_robustness_study import VARIANTS

START = pd.Timestamp("2026-07-01T00:00Z")


def candidate(symbol, minutes=0, asset="stock", **extra):
    entry = START+pd.Timedelta(minutes=minutes)
    return dict(symbol=symbol, asset=asset, entry_ts=entry,
                exit_ts=entry+pd.Timedelta(minutes=10), coverage_deadline=entry+pd.Timedelta(hours=1),
                gross=.01, funding=0., signal_strength=.1, side=1,
                signal_volume=2000., listing_days=10., hours_to_open=1., deviation=.06,
                entry_price=100., hold_h=1/6, **extra)


def test_crypto_signal_clock_is_not_execution_clock():
    row = dict(entry_ts=START, exit_ts=START+pd.Timedelta(hours=12, minutes=30))
    shifted = prepare_crypto(pd.DataFrame([row])).iloc[0]
    assert shifted.entry_ts == START+pd.Timedelta(minutes=15)
    assert shifted.coverage_deadline == row["exit_ts"]


def test_rejected_crypto_does_not_consume_cooldown():
    rows = [candidate("A"), candidate("B", asset="crypto", source_i=1, base_free_i=50),
            candidate("B", minutes=15, asset="crypto", source_i=2, base_free_i=51)]
    tr, _ = admit(pd.DataFrame(rows), VARIANTS[0], 68, START, START+pd.Timedelta(days=1), slots=1)
    assert tr.symbol.tolist() == ["A", "B"]
    assert tr.entry_ts.iloc[-1] == START+pd.Timedelta(minutes=15)


def test_sizing_changes_risk_not_admission_and_stock_daily_limit_remains():
    rows = [candidate("A", 0), candidate("B", 15), candidate("C", 30)]
    args = (pd.DataFrame(rows), replace(VARIANTS[0], stock_slots=1), 68, START, START+pd.Timedelta(days=1))
    small, _ = admit(*args, slots=1, weight=.2)
    large, _ = admit(*args, slots=1, weight=1.)
    assert small.symbol.tolist() == large.symbol.tolist() == ["A", "B"]
    assert small.notional.iloc[0] == .2 and large.notional.iloc[0] == 1.


def test_drawdown_recovery_distinguishes_unrecovered_losses():
    clock = pd.date_range(START, periods=5, freq="1D")
    done = drawdown_durations(pd.Series([1., .9, .8, .9, 1.], index=clock))
    assert done == dict(longest_drawdown_days=4., worst_drawdown_recovery_days=2., worst_drawdown_recovered=True)
    open_loss = drawdown_durations(pd.Series([1., .9, .8, .9, .95], index=clock))
    assert not open_loss["worst_drawdown_recovered"]
    assert np.isnan(open_loss["worst_drawdown_recovery_days"])


def test_crypto_mark_uses_only_completed_fifteen_minute_candles():
    clock = pd.date_range(START, periods=4, freq="15min")
    f = pd.DataFrame(dict(open=100., close=[100., 50., 100., 200.]), index=clock)
    trade = dict(asset="crypto", symbol="B", entry_ts=clock[1], exit_ts=clock[3],
                 entry_price=100., notional=.2, side=1, cost_bps=0., net=0.)
    curve = portfolio_curve(pd.DataFrame([trade]), {}, {"B": f}, START, clock[-1])
    assert curve.loc[clock[1]+pd.Timedelta(minutes=5)] == 1.
    assert curve.loc[clock[2]] == .9
    assert curve.iloc[-1] == 1.
