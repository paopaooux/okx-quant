import numpy as np
import pandas as pd
import pytest

from scripts.analysis.entry_rule_study import (
    Policy, equity_curve, filter_reason, first_run, path_exit, replay,
)


def path(prices):
    index = pd.date_range("2026-01-01", periods=len(prices), freq="5min", tz="UTC")
    return pd.DataFrame({k: prices for k in ("open", "high", "low", "close")}, index=index)


def candidate(symbol, minute=0, side=1, category="tech", early_minutes=5):
    start = pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(minutes=minute)
    return dict(symbol=symbol, entry_ts=start, side=side, category=category, asset="stock",
                strength=.02, entry_price=100., hours_to_open=5., volume_ratio=2., deviation=.02,
                base_exit_ts=start + pd.Timedelta(minutes=60), base_exit_price=100., base_reason="timeout",
                early_exit_ts=start + pd.Timedelta(minutes=early_minutes), early_exit_price=100.,
                early_reason="signal_invalid")


def test_three_complete_neutral_bars_and_next_open_execution():
    p = path([100., 101., 102., 103., 104.])
    result = path_exit(p, 1, .2, np.array([True, True, True, False, False]))
    assert result["early"]["exit_ts"] == p.index[3]
    assert result["early"]["exit_price"] == 103
    assert result["base"]["exit_ts"] == p.index[-1]
    assert first_run(np.array([True, True, False, True, True]), 3) is None


def test_stop_precedes_neutral_exit_and_ambiguous_target():
    p = path([100.] * 5)
    p.loc[p.index[2], ["low", "high"]] = [90, 110]
    result = path_exit(p, 1, .06, np.ones(5, dtype=bool))
    assert result["early"]["reason"] == "stop"
    assert result["early"]["exit_price"] == 94


def test_gap_stop_uses_worse_open():
    p = path([100., 110., 100.])
    assert path_exit(p, -1, .06, np.zeros(3, dtype=bool))["base"]["exit_price"] == 110


def test_early_exit_at_open_precedes_that_bars_extremes():
    p = path([100.] * 6)
    p.loc[p.index[3], "low"] = 80
    result = path_exit(p, 1, .06, np.ones(6, dtype=bool))
    assert result["early"]["reason"] == "signal_invalid"
    assert result["early"]["exit_price"] == 100


def test_released_slot_accepts_later_raw_candidate():
    rows = pd.DataFrame([candidate("A"), candidate("B", minute=5)])
    base, _ = replay(rows, Policy(slots=1), 0, 0)
    early, _ = replay(rows, Policy(slots=1, signal_exit=True), 0, 0)
    assert len(base) == 1
    assert len(early) == 2


def test_concentration_applies_across_assets_and_categories():
    rows = [candidate("A"), candidate("B"), candidate("C"),
            candidate("D", category="energy"), candidate("E", category="crypto")]
    rows[-1]["asset"] = "crypto"
    accepted, rejected = replay(pd.DataFrame(rows), Policy(concentration=True), 0, 0)
    assert accepted.symbol.tolist() == ["A", "B", "D"]
    assert rejected.reason.tolist() == ["category_limit", "direction_limit"]


def test_missing_volume_is_rejected_only_for_stock_filter():
    row = candidate("A")
    row["volume_ratio"] = np.nan
    assert filter_reason(row, Policy(entry_filter=True)) == "missing_volume"
    row["asset"] = "crypto"
    assert filter_reason(row, Policy(entry_filter=True)) is None


def test_rejected_candidate_does_not_block_later_signal():
    a = candidate("A")
    a["volume_ratio"] = 20
    accepted, _ = replay(pd.DataFrame([a, candidate("A", minute=5)]), Policy(entry_filter=True), 0, 0)
    assert len(accepted) == 1
    assert accepted.entry_ts.iloc[0].minute == 5


def test_open_position_drawdown_and_entry_fee_are_marked():
    p = path([100, 90, 100])
    row = candidate("A")
    row.update(base_exit_ts=p.index[2], base_exit_price=100.)
    trades, _ = replay(pd.DataFrame([row]), Policy(slots=1), 10, 10)
    days = pd.DatetimeIndex([p.index[0] - pd.Timedelta(minutes=5), p.index[2] - pd.Timedelta(seconds=1), p.index[2]])
    curve = equity_curve(trades, {"A": p}, days)
    assert curve.tolist() == pytest.approx([1, .8995, .999])


def test_future_prices_do_not_change_early_exit():
    p = path([100.] * 8)
    before = path_exit(p, 1, .06, np.ones(8, dtype=bool))["early"]
    p.loc[p.index[4]:, ["low", "close"]] = 70
    after = path_exit(p, 1, .06, np.ones(8, dtype=bool))["early"]
    assert before == after
