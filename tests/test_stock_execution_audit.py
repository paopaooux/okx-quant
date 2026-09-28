import numpy as np
import pandas as pd

from scripts.analysis.stock_execution_audit import STAGES, execution_stages


def test_execution_bridge_separates_entry_deadline_and_stop_gap():
    clock = pd.date_range("2026-07-01", periods=5, freq="5min", tz="UTC")
    f = pd.DataFrame({"open": [100, 101, 95, 99, 102], "high": [100, 102, 96, 101, 103],
                      "low": [100, 100, 94, 99, 102], "close": [100, 101, 95, 100, 103]}, index=clock)
    row = dict(symbol="A", entry_ts=clock[1], deadline=clock[4], signal_price=100.,
               entry_price=101., side=1, exit_price=95., exit_ts=clock[3],
               reason="stop", gross=95/101-1, hold_h=1/6)
    stages = execution_stages(pd.DataFrame([row]), {"A": f})
    assert stages[STAGES[0]].iloc[0].entry_price == 100
    assert stages[STAGES[1]].iloc[0].entry_price == 101
    assert np.isclose(stages[STAGES[2]].iloc[0].exit_price, 101*.97)
    assert stages[STAGES[3]].iloc[0].exit_price == 95


def test_deadline_open_never_uses_deadline_bar_extrema():
    clock = pd.date_range("2026-07-01", periods=4, freq="5min", tz="UTC")
    f = pd.DataFrame({"open": [100, 100, 100, 101], "high": [100, 101, 101, 200],
                      "low": [100, 99, 99, 1], "close": [100, 100, 100, 150]}, index=clock)
    row = dict(symbol="A", entry_ts=clock[1], deadline=clock[3], signal_price=100.,
               entry_price=100., side=1, exit_price=101., exit_ts=clock[3],
               reason="deadline", gross=.01, hold_h=1/6)
    stages = execution_stages(pd.DataFrame([row]), {"A": f})
    assert stages[STAGES[1]].iloc[0].exit_price == 100
    assert stages[STAGES[2]].iloc[0].exit_price == 101
