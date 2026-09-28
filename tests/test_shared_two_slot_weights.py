import numpy as np
import pandas as pd
import pytest

from scripts.analysis.shared_two_slot_weights import capital_usage, daily_risk, resize_book

START = pd.Timestamp("2026-07-01T00:00Z")


def row(symbol, entry, exit_, gross=.1, asset="stock"):
    return dict(symbol=symbol, asset=asset, entry_ts=START+pd.Timedelta(minutes=entry),
                exit_ts=START+pd.Timedelta(minutes=exit_), gross=gross, funding=0.,
                entry_price=100., notional=.2, side=1)


def test_reweight_compounds_closed_trades_not_open_profits():
    book = pd.DataFrame([row("A", 0, 10), row("B", 5, 15), row("C", 10, 20)])
    trades = resize_book(book, .3, 68)
    assert trades.notional.iloc[0] == trades.notional.iloc[1] == .3
    assert np.isclose(trades.notional.iloc[2], .3*(1+.3*(.1-.0068)))
    assert trades.position_weight.eq(.3).all()
    np.testing.assert_equal(trades.symbol.to_numpy(), book.symbol.to_numpy())


def test_cost_pair_and_funding_are_applied_before_resizing():
    book = pd.DataFrame([row("A", 0, 10), row("B", 10, 20, asset="crypto")])
    book.loc[0, "funding"] = .001
    trades = resize_book(book, .25, 44)
    np.testing.assert_allclose(trades.net, [.0966, .099])
    assert np.isclose(trades.notional.iloc[1], .25*(1+.25*.0966))


@pytest.mark.parametrize("weight", [0., -.2, .51, np.nan])
def test_invalid_weights_rejected(weight):
    with pytest.raises(ValueError):
        resize_book(pd.DataFrame([row("A", 0, 10)]), weight, 68)


def test_nonoverlap_and_solvent_sizing_are_required():
    with pytest.raises(ValueError):
        resize_book(pd.DataFrame([row("A", 0, 10), row("A", 5, 15)]), .4, 68)
    with pytest.raises(ValueError):
        resize_book(pd.DataFrame([row("A", 0, 10, gross=-3.), row("B", 10, 20)]), .4, 68)


def test_exposure_proxy_uses_completed_marks_and_reports_short_loss_risk():
    clock = pd.date_range(START, periods=4, freq="5min")
    f = pd.DataFrame(dict(open=100., close=[100., 120., 120., 120.]), index=clock)
    trades = pd.DataFrame([row("A", 5, 15)]).assign(notional=.8, side=-1)
    eq = pd.Series([1., 1., .84, .84], index=clock)
    detail, stats = capital_usage(trades, {"A": f}, {}, eq)
    assert detail.current_notional_ratio.iloc[1] == .8
    assert np.isclose(detail.current_notional_ratio.iloc[2], .8*1.2/.84)
    assert stats["negative_headroom_observations"] == 1
    assert stats["min_one_x_headroom_proxy_pct"] < 0


def test_daily_sharpe_keeps_idle_days_and_no_extra_terminal_zero():
    curve = pd.Series([1., 1.1, 1.1, .99], index=pd.date_range(START, periods=4, freq="1D"))
    returns, stats = daily_risk(curve)
    np.testing.assert_allclose(returns, [.1, 0., -.1], atol=1e-12)
    assert stats["sharpe_days"] == 3
    assert np.isclose(stats["sharpe_daily"], 0.)
    assert np.isclose(stats["daily_vol_pct"], 10.)


def test_daily_sharpe_uses_sample_deviation_and_365_day_annualization():
    curve = pd.Series([1., 1.1, 1.155], index=pd.date_range(START, periods=3, freq="1D"))
    _, stats = daily_risk(curve)
    assert np.isclose(stats["sharpe_daily"], .075/np.std([.1, .05], ddof=1)*np.sqrt(365))


def test_daily_sharpe_excludes_partial_days_and_uses_utc_boundaries():
    clock = pd.date_range(START+pd.Timedelta(hours=12), periods=49, freq="1h")
    curve = pd.Series(1., index=clock)
    curve.iloc[0], curve.iloc[12], curve.iloc[36], curve.iloc[-1] = .5, 1., 1.1, 2.
    curve.index = curve.index.tz_convert("Asia/Shanghai")
    returns, stats = daily_risk(curve)
    np.testing.assert_allclose(returns, [.1])
    assert stats["sharpe_days"] == 1
    assert np.isnan(stats["sharpe_daily"])


def test_initial_midnight_entry_cost_is_not_lost():
    curve = pd.Series([.99, 1., 1.1], index=pd.date_range(START, periods=3, freq="1D"))
    returns, _ = daily_risk(curve)
    np.testing.assert_allclose(returns, [0., .1])


def test_flat_daily_equity_has_undefined_sharpe():
    _, stats = daily_risk(pd.Series(1., index=pd.date_range(START, periods=4, freq="1D")))
    assert stats["sharpe_days"] == 3
    assert np.isnan(stats["sharpe_daily"])
    assert stats["annualized_vol_pct"] == 0.


def test_missing_daily_mark_fails_instead_of_filling():
    curve = pd.Series([1., 1.1], index=[START, START+pd.Timedelta(days=2)])
    with pytest.raises(ValueError, match="Missing midnight"):
        daily_risk(curve)


@pytest.mark.parametrize("values", [[1., 0.], [1., np.nan], [1., np.inf]])
def test_invalid_equity_cannot_produce_sharpe(values):
    with pytest.raises(ValueError):
        daily_risk(pd.Series(values, index=pd.date_range(START, periods=2, freq="1D")))
