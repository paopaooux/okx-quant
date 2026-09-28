import pandas as pd

from scripts.analysis.reference_calendar_full_history import remove_gap_affected


def test_gap_between_anchor_and_exit_is_excluded_before_allocation():
    index = pd.date_range("2026-03-04T20:00Z", periods=8, freq="5min")
    frames = {"TEST": pd.DataFrame({"close": 100.}, index=index.delete(4))}
    events = pd.DataFrame({"inst_id": ["TEST", "TEST"],
                           "event_ts": [index[1], index[5]],
                           "anchor_close_ts": [index[0], index[0]]})
    candidates = pd.DataFrame({"inst_id": ["TEST", "TEST"],
                               "entry_ts": [index[1], index[5]],
                               "exit_ts": [index[2], index[7]]})
    accepted, rejected = remove_gap_affected(events, candidates, frames)
    assert accepted.entry_ts.tolist() == [index[1]]
    assert rejected.entry_ts.tolist() == [index[5]]


def test_gap_after_exit_does_not_reject_a_valid_trade():
    index = pd.date_range("2026-03-04T20:00Z", periods=8, freq="5min")
    frames = {"TEST": pd.DataFrame({"close": 100.}, index=index.delete(7))}
    events = pd.DataFrame({"inst_id": ["TEST"], "event_ts": [index[1]],
                           "anchor_close_ts": [index[0]]})
    candidates = pd.DataFrame({"inst_id": ["TEST"], "entry_ts": [index[1]], "exit_ts": [index[2]]})
    accepted, rejected = remove_gap_affected(events, candidates, frames)
    assert len(accepted) == 1 and rejected.empty
