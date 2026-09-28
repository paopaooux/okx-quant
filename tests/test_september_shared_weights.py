import numpy as np
import pandas as pd
import pytest

from scripts.analysis.september_shared_weights import crypto_candidates
from scripts.analysis.stock_portfolio_robustness import admit
from scripts.analysis.stock_robustness_study import VARIANTS

START = pd.Timestamp("2026-09-04T00:00Z")
END = START+pd.Timedelta(days=2)


def predictions():
    return pd.DataFrame([dict(symbol="A", bar_at=START, observed_at=START+pd.Timedelta(minutes=16),
        signal_side="long", width=.1, p_up=.6)])


def market():
    return pd.DataFrame(dict(open=100., high=101., low=99., close=100.),
        index=pd.date_range(START-pd.Timedelta(minutes=15), END, freq="15min"))


def test_recorded_direction_is_not_rethresholded_and_entry_is_causal():
    data = predictions()
    data.loc[0, "p_up"] = .51
    tr, excluded = crypto_candidates(data, {"A": market()}, START, END)
    assert len(tr) == 1 and excluded.empty
    assert tr.entry_ts.iloc[0] == START+pd.Timedelta(minutes=30)
    assert tr.entry_ts.iloc[0] >= tr.observed_at.iloc[0]
    assert tr.coverage_deadline.iloc[0] == START+pd.Timedelta(minutes=49*15)
    data.loc[0, ["signal_side", "p_up"]] = ["flat", .99]
    tr, _ = crypto_candidates(data, {"A": market()}, START, END)
    assert tr.empty


def test_duplicate_bar_revisions_must_not_be_selected_with_hindsight():
    data = predictions()
    with pytest.raises(ValueError, match="first-observation"):
        crypto_candidates(pd.concat([data, data]), {"A": market()}, START, END)


@pytest.mark.parametrize("field,value,reason", [
    ("width", np.nan, "invalid_width"),
    ("width", 0., "invalid_width"),
    ("p_up", np.nan, "invalid_probability"),
    ("observed_at", START+pd.Timedelta(minutes=14), "unclosed_or_stale_bar"),
    ("observed_at", START+pd.Timedelta(minutes=31), "unclosed_or_stale_bar"),
])
def test_invalid_live_candidates_are_excluded(field, value, reason):
    data = predictions()
    data.loc[0, field] = value
    tr, rejected = crypto_candidates(data, {"A": market()}, START, END)
    assert tr.empty and rejected.reason.tolist() == [reason]


def test_complete_known_horizon_required_even_for_early_winner():
    prices = market()
    prices.loc[START+pd.Timedelta(minutes=30), "high"] = 120.
    tr, rejected = crypto_candidates(predictions(), {"A": prices}, START, START+pd.Timedelta(hours=1))
    assert tr.empty and rejected.reason.tolist() == ["outside_known_horizon"]
    missing = prices.drop(START+pd.Timedelta(hours=10))
    tr, rejected = crypto_candidates(predictions(), {"A": missing}, START, END)
    assert tr.empty and rejected.reason.tolist() == ["incomplete_market_window"]


def test_shared_rejection_keeps_later_live_observations_available():
    data = predictions()
    next_bar = data.copy()
    next_bar["bar_at"] += pd.Timedelta(minutes=15)
    next_bar["observed_at"] += pd.Timedelta(minutes=15)
    tr, _ = crypto_candidates(pd.concat([data, next_bar]), {"A": market()}, START, END)
    assert len(tr) == 2  # Never preselect just the first crypto-only position.
    blocker = tr.iloc[[0]].copy()
    blocker["symbol"] = "B"
    blocker["entry_ts"] = START
    blocker["exit_ts"] = START+pd.Timedelta(minutes=40)
    book, _ = admit(pd.concat([blocker, tr]), VARIANTS[0], 68, START, END, slots=1)
    assert book.symbol.tolist() == ["B", "A"]
    assert book.entry_ts.iloc[-1] == START+pd.Timedelta(minutes=45)


def test_no_extra_frozen_label_cooldown_after_a_completed_live_trade():
    data = predictions()
    next_bar = data.copy()
    next_bar["bar_at"] += pd.Timedelta(minutes=15)
    next_bar["observed_at"] += pd.Timedelta(minutes=15)
    prices = market()
    prices.loc[START+pd.Timedelta(minutes=30), "high"] = 120.
    tr, _ = crypto_candidates(pd.concat([data, next_bar]), {"A": prices}, START, END)
    book, _ = admit(tr, VARIANTS[0], 68, START, END, slots=1)
    assert len(book) == 2
    assert book.exit_ts.iloc[0] == book.entry_ts.iloc[1]
