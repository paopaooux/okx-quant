from types import SimpleNamespace

import pandas as pd
import pytest

from scripts.analysis.stock_live_diagnosis import earlier_exit, product_group


def fixture():
    start = pd.Timestamp("2026-09-01T00:00:00Z")
    row = SimpleNamespace(cTime=start, uTime=start+pd.Timedelta(hours=12),
                          direction="long", notional=100, openAvgPx=100, closeAvgPx=102,
                          fee=-.101, realizedPnl=1.899, instId="TEST")
    desired = start+pd.Timedelta(hours=6)
    market = pd.DataFrame({"open": [101.]}, index=[desired])
    funding = pd.DataFrame(dict(inst=["TEST", "TEST"],
        ts=[start+pd.Timedelta(hours=4), start+pd.Timedelta(hours=8)], amount=[-.01, -.02]))
    return row, desired, market, funding


def test_earlier_exit_uses_only_funding_while_held():
    row, desired, market, funding = fixture()
    result = earlier_exit(row, desired, market, funding, 0)
    assert result["net"] == pytest.approx(1-.1005-.01)
    assert result["changed"] and not result["missing"]


def test_actual_earlier_stop_is_preserved():
    row, desired, market, funding = fixture()
    row.uTime = row.cTime+pd.Timedelta(hours=2)
    result = earlier_exit(row, desired, market, funding, 30)
    assert result["net"] == row.realizedPnl and not result["changed"]


def test_missing_price_not_forward_filled():
    row, desired, market, funding = fixture()
    result = earlier_exit(row, desired+pd.Timedelta(minutes=5), market, funding, 0)
    assert result["missing"] and pd.isna(result["net"])


@pytest.mark.parametrize("direction", ["long", "short"])
def test_slippage_always_hurts(direction):
    row, desired, market, funding = fixture()
    row.direction = direction
    assert earlier_exit(row, desired, market, funding, 10)["net"] < earlier_exit(
        row, desired, market, funding, 0)["net"]


def test_exit_cannot_precede_entry():
    row, _, market, funding = fixture()
    with pytest.raises(ValueError):
        earlier_exit(row, row.cTime, market, funding, 0)


def test_identity_classification_keeps_unknowns_explicit():
    assert product_group("INTW-USDT-SWAP") == "leveraged_inverse"
    assert product_group("MINIMAX-USDT-SWAP") == "hong_kong_equity"
    assert product_group("SHEIN-USDT-SWAP") == "other_identity_pending"
    assert product_group("UNKNOWN-USDT-SWAP") == "unclassified"
