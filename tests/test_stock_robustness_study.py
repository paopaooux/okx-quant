from dataclasses import replace

import numpy as np
import pandas as pd

from scripts.analysis.stock_robustness_study import (
    STEP, Variant, add_event_features, choose_rule, entry_filter, marked_curve,
    price_candidates, replay,
)
from scripts.analysis.strategy_replacement_study import path_exit


def frame():
    index = pd.date_range("2026-06-01T20:00Z", periods=40, freq=STEP)
    return pd.DataFrame(dict(open=100., high=101., low=99., close=100., volume_quote=2000.), index=index)


def event(f):
    return dict(inst_id="TEST", event_ts=f.index[2], resolve_ts=f.index[14], side=1, deviation=-.06,
                anchor_close_ts=f.index[0]+STEP, signal_price=100., anchor_price=100/.94,
                signal_volume=2000., volume_24h=200000., listing_days=10., hours_to_open=1.,
                window_hours=16., coverage_deadline=f.index[26])


def test_five_minute_scanner_preserves_stop_priority_and_gap_fill():
    f = frame()
    f.loc[f.index[3], ["open", "low", "high"]] = [95., 94., 105.]
    result = path_exit(f, f.index[2], 1, .03, .03, f.index[5], step=STEP)
    assert result == (100., 95., f.index[4], "stop")


def test_next_open_not_signal_close_and_no_pre_entry_extrema():
    f = frame()
    f.loc[f.index[1], "low"] = 1.
    f.loc[f.index[2], ["open", "high"]] = [102., 103.]
    candidates, excluded = price_candidates(pd.DataFrame([event(f)]), {"TEST": f}, Variant("test"), {})
    assert excluded.empty
    assert candidates.iloc[0].entry_price == 102.
    assert candidates.iloc[0].reason == "deadline"


def test_delay_cannot_enter_after_reference_open():
    f = frame()
    e = event(f)
    e["resolve_ts"] = e["event_ts"]+STEP
    candidates, excluded = price_candidates(pd.DataFrame([e]), {"TEST": f}, Variant("test", delay_bars=1), {})
    assert candidates.empty
    assert excluded.iloc[0].reason == "expired"


def test_common_preknown_deadline_purges_even_early_winning_exit():
    f = frame()
    candidates, _ = price_candidates(pd.DataFrame([event(f)]), {"TEST": f}, Variant("test"), {})
    candidates["exit_ts"] = f.index[3]
    candidates["gross"] = .1
    trades, _ = replay(candidates, Variant("test"), 44, f.index[0], f.index[10])
    assert trades.empty


def test_entry_rules_use_pretrade_information_only():
    row = event(frame()) | dict(remaining_bps=610., adverse_bps=5.)
    v = Variant("test", entry_guard=True, volume_24h=150000)
    assert entry_filter(row, v) is None
    assert entry_filter(row | dict(gross=-.9), v) is None
    assert entry_filter(row | dict(remaining_bps=590.), v) == "fresh_price"
    assert entry_filter(row | dict(volume_24h=np.nan), v) == "volume_24h"


def test_features_are_unchanged_by_future_prices_or_volume():
    f = frame()
    e = pd.DataFrame([event(f)])
    u = pd.DataFrame([dict(instId="TEST", list_ts="2026-05-01T00:00Z")])
    before = add_event_features(e, {"TEST": f}, u)
    f.loc[f.index[2]:, ["close", "volume_quote"]] = [999., 999999.]
    after = add_event_features(e, {"TEST": f}, u)
    pd.testing.assert_frame_equal(before, after)


def test_marked_drawdown_sees_unrealized_loss():
    f = frame()
    f.loc[f.index[3], "close"] = 90.
    trades = pd.DataFrame([dict(symbol="TEST", entry_ts=f.index[2], exit_ts=f.index[5],
        entry_price=100., side=1, notional=.2, cost_bps=44., net=-.0044)])
    eq = marked_curve(trades, {"TEST": f}, f.index[0], f.index[8])
    assert eq.loc[f.index[4]] < .98
    assert np.isclose(eq.iloc[-1], 1-.2*.0044)


def test_selection_ignores_diagnostics_and_has_cash_option():
    common = dict(trades=50, entry_days=20, mean_net_bps=50., marked_dd_pct=-3.)
    baseline = pd.Series(common | dict(variant="baseline", return_pct=6.))
    rows = pd.DataFrame([baseline.to_dict(), common | dict(variant="liquidity", return_pct=8.),
                         common | dict(variant="long_only", return_pct=1000.)])
    assert choose_rule(rows, baseline) == "liquidity"
    rows["mean_net_bps"] = -1.
    assert choose_rule(rows, baseline) == "cash"
