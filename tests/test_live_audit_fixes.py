import json
from decimal import Decimal
from unittest.mock import Mock

import pandas as pd
import pytest

from scripts.live import auto_demo as live, okx_demo as api, stock_data_loop as feed
from scripts.live.combination_policy import entry_rejection
from test_live_combination_policy import NOW, position, signal, state
from test_live_execution_recovery import ledger


def balance(eq, cash, upl="0"):
    return [{"totalEq": "10000", "details": [{"ccy": "USDT", "eq": str(eq),
             "cashBal": str(cash), "availBal": str(cash), "upl": upl}]}]


@pytest.mark.parametrize("weight,count,expected,remaining", [
    (.2, 5, [20, 20, 20, 20, 19], 1), (.35, 2, [35, 35], 30)])
def test_isolated_slots_keep_equity_base_and_reserve_once(monkeypatch, weight, count, expected, remaining):
    monkeypatch.setattr(live, "SLOT_WEIGHT", weight)
    specs = {str(i): {"ctVal": "1", "ctValCcy": "USDT", "lotSz": ".01", "minSz": ".01"}
             for i in range(5)}
    s = state()
    cash = Decimal("100")
    sizes = []
    for i in range(count):
        equity, budget = live.entry_budget(balance(100, cash), s, specs, {})
        assert equity == 100
        size = live.size_for_signal(None, {"inst_id": str(i)}, 1, equity, specs, budget=budget)
        sizes.append(Decimal(size))
        cash -= Decimal(size)
        s["positions"][str(i)] = dict(inst_id=str(i), size=size, entry_px=1)
    assert sizes == expected
    assert cash == remaining


@pytest.mark.parametrize("eq,upl,expected", [(110, "10", 100), (90, "-10", 90)])
def test_no_unrealized_profit_or_foreign_collateral_in_sizing(eq, upl, expected):
    equity, _ = live.entry_budget(balance(eq, 80, upl), state(), {}, {})
    assert equity == expected


def test_partial_entry_reservation_is_not_counted_twice():
    s = state({"a": dict(inst_id="A", size="10", entry_px=1)})
    s["pending_entries"] = {"id": dict(inst_id="A", size="20", entry_px=1)}
    specs = {"A": {"ctVal": "1", "ctValCcy": "USDT"}}
    assert live.entry_budget(balance(100, 90), s, specs, {}) == (100, 79)


@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("pending", [False, True])
def test_underlying_limit_includes_pending_and_inverse_products(side, pending):
    p = position("SKDD-USDT-SWAP")
    p["side"] = side
    s = state() if pending else state({"a": p})
    if pending:
        s["pending_entries"]["a"] = p
    assert entry_rejection(s, signal("SKUU-USDT-SWAP"), NOW) == "same_underlying"
    assert entry_rejection(s, signal("CSOPSKHYNIX2L-USDT-SWAP"), NOW) == "same_underlying"
    assert entry_rejection(s, signal("UNRELATED-USDT-SWAP"), NOW) is None


@pytest.mark.parametrize("pnl", ["-.3", ".2", "0"])
def test_isolated_funding_replay_and_legacy_stats(ledger, capsys, pnl):
    bill = dict(billId="a", ts="1790000000000", type="8", subType="173",
                ccy="USDT", balChg="0", pnl=pnl, posBalChg=pnl)
    api.save_account_data([], [], [bill])
    with api.db_connect() as db:
        assert db.execute("SELECT amount FROM bills").fetchone()[0] == pnl
        db.execute("UPDATE bills SET amount='0'")
    api.print_stats()
    assert f"funding_cashflow={float(pnl):.8f}" in capsys.readouterr().out
    api.save_account_data([], [], [bill])
    with api.db_connect() as db:
        assert db.execute("SELECT amount FROM bills").fetchone()[0] == pnl
        assert db.execute("SELECT count(*) FROM bills").fetchone()[0] == 1


def test_funding_fallback_does_not_sum_duplicate_balance_fields():
    assert api.bill_amount(dict(type="8", pnl="", posBalChg="-.2", balChg="-.2")) == "-.2"
    assert api.bill_amount(dict(type="8", balChg=".3")) == ".3"
    assert api.bill_amount(dict(type="2", pnl="99", balChg="1")) == "1"


def test_feed_publishes_before_whole_batch_finishes(tmp_path, monkeypatch):
    path = tmp_path / "universe.csv"
    pd.DataFrame({"instId": [f"A{i:02}" for i in range(12)]}).to_csv(path, index=False)
    monkeypatch.setattr(feed, "UNIVERSE", path)
    seen = []
    def update(*args):
        if len(seen) == 11:
            assert progress == [1, 10]
        seen.append(args[1])
    progress = []
    monkeypatch.setattr(feed, "update_cache", update)
    assert feed.refresh_candles(Mock(), progress.append) == 12
    assert progress == [1, 10]


def test_partial_feed_failure_does_not_hide_successful_updates(tmp_path, monkeypatch):
    monkeypatch.setattr(feed, "STATUS", tmp_path / "status.json")
    client = Mock()
    monkeypatch.setattr(feed, "OKXClient", lambda **kwargs: client)
    def refresh(client, on_progress):
        on_progress(1)
        published = json.loads(feed.STATUS.read_text())
        assert published["candles"]["refreshing"]
        assert published["candles"]["last_success_at"]
        raise RuntimeError("outage")
    monkeypatch.setattr(feed, "refresh_candles", refresh)
    result = feed.run_once()
    assert result["candles"]["last_error"] == "outage"
    assert not result["candles"]["refreshing"]
    client.session.close.assert_called_once()


def test_refresh_is_aligned_after_clock_boundary(monkeypatch):
    monkeypatch.setattr(feed, "INTERVAL", 60)
    assert feed.next_refresh_delay(299) == 3
    assert feed.next_refresh_delay(301) == 1
    assert feed.next_refresh_delay(302) == 60
