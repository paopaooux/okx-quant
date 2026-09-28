import pandas as pd
import pytest

from scripts.analysis.live_month_replay import STEP, replay, replay_exit


def market():
    index = pd.date_range("2026-09-01T00:00:00Z", periods=50, freq=STEP)
    return pd.DataFrame(dict(open=100., high=101., low=99., close=100.), index=index)


def test_same_bar_stop_and_target_uses_stop_first():
    f = market()
    f.loc[f.index[0], ["high", "low"]] = [110, 90]
    result = replay_exit(f, f.index[0], f.index[48], 1, .03)
    assert result["reason"] == "stop" and result["ambiguous"]
    assert result["exit_px"] == 97
    assert result["exit_ts"] == f.index[1]


@pytest.mark.parametrize("side,opening,price", [(1, 90, 90), (-1, 110, 110)])
def test_stop_gap_execution(side, opening, price):
    f = market()
    f.loc[f.index[1], ["open", "high", "low"]] = [opening, opening+1, opening-1]
    result = replay_exit(f, f.index[0], f.index[48], side, .03)
    assert result["exit_px"] == price


def test_missing_interior_bar_is_not_silently_skipped():
    f = market()
    assert replay_exit(f.drop(f.index[7]), f.index[0], f.index[48], 1, .03) is None


def test_deadline_does_not_use_later_extrema():
    f = market()
    f.loc[f.index[48], ["high", "low"]] = [200, 1]
    result = replay_exit(f, f.index[0], f.index[48], 1, .03)
    assert result["reason"] == "deadline" and result["exit_px"] == 100


def test_position_occupancy_and_costs():
    f = market()
    rows = []
    for i in range(2):
        rows.append(dict(symbol="BTCUSDT", observed_at=f.index[i],
                         entry_ts=f.index[i], deadline=f.index[48+i],
                         side=1, signal_side="long", p_up=.6, width=.03, age_s=1000))
    trades, rejected = replay(pd.DataFrame(rows), {"BTCUSDT": f}, "recorded", 10)
    assert len(trades) == 1 and rejected.reason.tolist() == ["same_instrument"]
    assert trades.iloc[0].after_cost_return < -.0029
