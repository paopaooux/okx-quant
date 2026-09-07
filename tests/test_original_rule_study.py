import numpy as np
import pandas as pd
import pytest

from scripts.analysis.entry_rule_study import Policy
from scripts.analysis.original_rule_study import START, marked_curve, neutral_exit, pooled, sleeve_select
from scripts.combinations.run import _pooled_curve


def bars():
    index = pd.date_range(START, periods=9, freq="5min")
    return pd.DataFrame({"open": np.arange(100., 109.)}, index=index)


def candidate(symbol, minutes=0, category="tech", side=1):
    stamp = START + pd.Timedelta(minutes=minutes)
    return dict(symbol=symbol, category=category, side=side, asset="stock", strategy="xstock_hybrid",
                entry_ts=stamp, exit_ts=stamp + pd.Timedelta(hours=2),
                original_exit_ts=stamp + pd.Timedelta(hours=2), gross=.02, net=.0156,
                early_exit_ts=stamp + pd.Timedelta(minutes=5), early_gross=.01,
                early_exit_price=101., early_reason="signal_invalid", reason="deadline",
                signal_strength=.065, deviation=.065, hours_to_open=5., volume_ratio=2.)


def test_neutral_exit_uses_next_open_not_future_close():
    frame = bars()
    result = neutral_exit(frame, np.ones(len(frame), dtype=bool), START, frame.index[-1], pd.Timedelta(minutes=5))
    assert result == (frame.index[3], 104.)
    frame.iloc[5:, 0] = 500
    assert neutral_exit(frame, np.ones(len(frame), dtype=bool), START, frame.index[-1], pd.Timedelta(minutes=5)) == result


def test_missing_bar_resets_consecutive_neutral_count():
    frame = bars().drop(bars().index[2])
    stamp, price = neutral_exit(frame, np.ones(len(frame), dtype=bool), START, frame.index[-1], pd.Timedelta(minutes=5))
    assert stamp == START + pd.Timedelta(minutes=25)
    assert price == 106


def test_equal_timestamp_preserves_prior_stop():
    frame = bars()
    assert neutral_exit(frame, np.ones(len(frame), dtype=bool), START, frame.index[3], pd.Timedelta(minutes=5)) is None


def test_early_exit_releases_same_symbol_but_preserves_daily_cap():
    candidates = pd.DataFrame([candidate("A"), candidate("A", 10), candidate("B", 20)])
    baseline = sleeve_select(candidates, Policy())
    early = sleeve_select(candidates, Policy(signal_exit=True))
    assert baseline.symbol.tolist() == ["A", "B"]
    assert early.symbol.tolist() == ["A", "A"]
    assert early.reason.eq("signal_invalid").all()


def test_entry_filter_runs_before_daily_admission():
    first = candidate("A")
    first["volume_ratio"] = np.nan
    selected = sleeve_select(pd.DataFrame([first, candidate("B", 5), candidate("C", 10)]), Policy(entry_filter=True))
    assert selected.symbol.tolist() == ["B", "C"]


def test_pool_matches_original_engine_with_flags_off():
    trades = pd.DataFrame([candidate("A"), candidate("B", 180), candidate("C", 180, side=-1)])
    days = pd.date_range(START.normalize(), periods=3, freq="D")
    curve, selected, _ = pooled(trades, Policy(), days)
    expected, original, _ = _pooled_curve(trades, days, 5)
    np.testing.assert_allclose(curve, expected)
    assert selected.symbol.tolist() == original.symbol.tolist()
    assert curve.iloc[-1] == pytest.approx(1.00312 * 1.00624)


def test_shared_concentration_crosses_asset_boundary():
    rows = [candidate("A"), candidate("B"), candidate("C"),
            candidate("D", category="energy"), candidate("E", category="crypto")]
    rows[-1]["asset"] = "crypto"
    _, selected, rejected = pooled(pd.DataFrame(rows), Policy(concentration=True))
    assert selected.symbol.tolist() == ["A", "B", "D"]
    assert rejected.reason.tolist() == ["category_limit", "direction_limit"]


def test_marked_curve_includes_open_loss_and_half_cost():
    row = candidate("A")
    row.update(entry_price=100., notional=.2, exit_ts=START + pd.Timedelta(minutes=10), gross=0., net=-.004)
    cutoffs = pd.date_range(START - pd.Timedelta(minutes=5), periods=4, freq="5min")
    frame = pd.DataFrame({"close": [100., 100., 90., 100.]}, index=cutoffs)
    curve = marked_curve(pd.DataFrame([row]), {"A": frame}, cutoffs)
    assert curve.tolist() == pytest.approx([1., .9996, .9796, .9992])
