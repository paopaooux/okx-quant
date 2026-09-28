import pandas as pd
import pytest

from scripts.analysis import us_only_study as study
from scripts.analysis.entry_rule_study import Policy
from scripts.analysis.original_rule_study import sleeve_select


@pytest.mark.parametrize("name,etf,expected", [
    ("Company - Common Stock", "N", True),
    ("Company - American Depositary Shares", "N", True),
    ("Company - New York Registry Shares", "N", True),
    ("Company - Class A Subordinate Voting Shares", "N", True),
    ("Company - Preferred Stock", "N", False),
    ("Company - Depositary Shares of Preferred Stock", "N", False),
    ("Common Stock ETF", "Y", False),
    ("Company - Units", "N", False),
])
def test_security_type(name, etf, expected):
    assert study.is_ordinary(name, etf) is expected


def test_filter_before_admission_allows_replacement():
    start = pd.Timestamp("2026-04-01T00:00:00Z")
    candidates = pd.DataFrame([dict(entry_ts=start+pd.Timedelta(minutes=i),
        exit_ts=start+pd.Timedelta(hours=12), original_exit_ts=start+pd.Timedelta(hours=12),
        symbol=symbol, asset="stock", signal_strength=1)
        for i, symbol in enumerate(["FOREIGN", "US1", "US2"])])
    assert sleeve_select(candidates, Policy()).symbol.tolist() == ["FOREIGN", "US1"]
    filtered = sleeve_select(candidates[candidates.symbol.ne("FOREIGN")], Policy())
    assert filtered.symbol.tolist() == ["US1", "US2"]
