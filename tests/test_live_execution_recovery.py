import copy
import json
import threading
import time
from decimal import Decimal
from unittest.mock import Mock

import pandas as pd
import pytest
import requests

from scripts.live import auto_demo as live, execution_risk as risk, okx_demo as api, sync_loop
from test_live_combination_policy import FakeClient, loop, position, signal, state


def test_definitive_exit_rejection_retries_after_backoff(loop):
    client, now = loop
    p = position(now=now - pd.Timedelta(hours=4))
    s = state({"TEST|long": p})
    client.order = Mock(side_effect=[api.OKXAPIError("51008", "insufficient balance"), [{"ordId": "ok"}]])
    live.manage_positions(client, s, live.LiveData(None), now, True)
    assert "exit_client_id" not in p and p["exit_retry_after"] > time.time()
    live.manage_positions(client, s, live.LiveData(None), now, True)
    assert client.order.call_count == 1
    p["exit_retry_after"] = 0
    live.manage_positions(client, s, live.LiveData(None), now, True)
    assert client.order.call_count == 2 and not s["positions"]


@pytest.mark.parametrize("failure", [requests.Timeout(), api.OKXAPIError("50004", "timeout"),
                                      api.OKXAPIError("51016", "duplicate id")])
def test_ambiguous_exit_keeps_identity_and_never_resubmits(loop, failure):
    client, now = loop
    p = position(now=now - pd.Timedelta(hours=4))
    s = state({"TEST|long": p})
    client.order = Mock(side_effect=failure)
    client.order_detail = Mock(side_effect=api.OKXAPIError("51603", "not found"))
    live.manage_positions(client, s, live.LiveData(None), now, True)
    cid = p["exit_client_id"]
    live.manage_positions(client, s, live.LiveData(None), now, True)
    assert client.order.call_count == 1 and p["exit_client_id"] == cid


def test_rejected_entry_releases_quota_and_retries_same_signal(loop, monkeypatch):
    client, now = loop
    sig = signal(now=now)
    s = state()
    monkeypatch.setattr(live, "build_stock_signals", lambda _: {sig["inst_id"]: sig})
    client.order = Mock(side_effect=[api.OKXAPIError("51008"), [{"ordId": "ok"}]])
    live.run_once(client, s, True)
    assert not s["pending_entries"] and s["stock_entries_by_day"][str(now.date())] == 0
    key = sig["inst_id"] + "|long"
    assert key not in s["last_bars"]
    live.run_once(client, s, True)
    assert client.order.call_count == 1
    s["entry_retries"][key]["retry_after"] = 0
    live.run_once(client, s, True)
    assert client.order.call_count == 2 and s["stock_entries_by_day"][str(now.date())] == 1


def test_unknown_entry_requires_expiry_three_absences_and_no_remote_exposure():
    client = Mock()
    client.pending_orders.return_value = []
    client.positions.return_value = []
    p = {"inst_id": "TEST", "exp_time": int(time.time() * 1000) - 120_000}
    error = api.OKXAPIError("51603")
    assert not risk.definitely_absent(client, {}, "id", error)
    assert not risk.definitely_absent(client, p, "id", requests.Timeout())
    assert not risk.definitely_absent(client, p, "id", error)
    assert not risk.definitely_absent(client, p, "id", error)
    client.positions.return_value = [{"instId": "TEST", "pos": "1"}]
    assert not risk.definitely_absent(client, p, "id", error)
    client.positions.return_value = []
    assert risk.definitely_absent(client, p, "id", error)


def test_missing_expired_exit_uses_current_remote_size():
    c = Mock()
    c.pending_orders.return_value = []
    c.positions.return_value = [{"instId": "TEST", "pos": "0.3", "mgnMode": "isolated"}]
    p = dict(inst_id="TEST", side="long", size="1", exit_client_id="old",
             exit_exp_time=int(time.time() * 1000) - 120_000)
    for i in range(3):
        assert risk.recover_missing_exit(c, p, api.OKXAPIError("51603")) is (i == 2)
    assert p["size"] == "0.3" and "exit_client_id" not in p


def test_expired_exit_does_not_fetch_price(loop):
    client, now = loop
    s = state({"p": position(now=now - pd.Timedelta(hours=4))})
    data = Mock()
    data.ticker.side_effect = AssertionError("deadline must not depend on ticker")
    live.manage_positions(client, s, data, now, True)
    assert len(client.orders) == 1 and not s["positions"]


def test_account_failure_still_closes_local_deadline_and_blocks_entries(loop, monkeypatch):
    client, now = loop
    s = state({"TEST|long": position(now=now - pd.Timedelta(hours=4))})
    monkeypatch.setattr(live, "remote_positions", Mock(side_effect=requests.Timeout()))
    live.run_once(client, s, True, prepared_signals={"NEW": signal("NEW", now=now)})
    assert len(client.orders) == 1 and client.orders[0][3][1] is True
    assert not s["entry_reconciliation_ok"]


def test_signal_worker_does_not_block_exit_execution(loop, monkeypatch):
    client, now = loop
    started, release = threading.Event(), threading.Event()
    def blocked(_):
        started.set()
        assert release.wait(5)
        return {}
    monkeypatch.setattr(live, "collect_signals", blocked)
    worker = live.SignalWorker(None)
    p = position(now=now - pd.Timedelta(hours=4))
    s = state({"TEST|long": p})
    monkeypatch.setattr(live, "remote_positions", lambda _: {"TEST|long": {**p, "upl": 0}})
    try:
        assert worker.poll() is None
        assert started.wait(2)
        live.run_once(client, s, True, prepared_signals={}, management_only=True)
        assert len(client.orders) == 1 and not release.is_set()
    finally:
        release.set()
        worker.close()


def test_protective_stop_survives_restart_without_duplicate(loop):
    client, now = loop
    s = state({"TEST|long": position(now=now)})
    client.stop_order = Mock(return_value=[{"algoId": "stop"}])
    client.algo_detail = Mock(return_value={"algoId": "stop", "state": "live"})
    specs = {"TEST-USDT-SWAP": {"tickSz": ".1"}}
    snapshots = []
    risk.protect_positions(client, s, specs, lambda x: snapshots.append(copy.deepcopy(x)))
    assert snapshots[0]["protective_orders"]
    assert client.stop_order.call_args.args[2] == "97.0"
    restored = json.loads(json.dumps(s))
    for record in restored["protective_orders"].values():
        record["check_after"] = 0
    risk.protect_positions(client, restored, specs, lambda _: None)
    assert client.stop_order.call_count == 1


def test_protection_failure_forces_exit_without_duplicate_stop(loop):
    client, now = loop
    s = state({"p": position(now=now)})
    client.stop_order = Mock(side_effect=requests.Timeout())
    risk.protect_positions(client, s, {"TEST-USDT-SWAP": {"tickSz": ".01"}}, lambda _: None)
    assert s["positions"]["p"]["force_exit"] == "protection_unavailable"
    risk.protect_positions(client, s, {}, lambda _: None)
    assert client.stop_order.call_count == 1
    live.manage_positions(client, s, live.LiveData(None), now, True)
    assert not s["positions"] and s["protective_orders"]


def test_orphan_stop_cancellation_waits_for_readback():
    s = state()
    s["protective_orders"] = {"stop": {"inst_id": "TEST", "position_key": "TEST|long"}}
    c = Mock()
    c.algo_detail.side_effect = [{"state": "live", "algoId": "a"}, {"state": "canceled", "algoId": "a"}]
    risk.cleanup_protection(c, s, lambda _: None)
    assert s["protective_orders"]
    risk.cleanup_protection(c, s, lambda _: None)
    assert not s["protective_orders"] and c.cancel_algo.call_count == 1


def test_stop_order_is_full_position_reduce_only():
    c = object.__new__(api.DemoClient)
    c._request = Mock(return_value=[{"algoId": "a"}])
    c.stop_order("BTC-USDT-SWAP", "sell", "100", "sl1")
    body = c._request.call_args.kwargs["body"]
    assert body["reduceOnly"] is True and body["closeFraction"] == "1" and "sz" not in body
    assert body["posSide"] == "net" and body["slOrdPx"] == "-1"


@pytest.mark.parametrize("side,expected", [("long", "97.0"), ("short", "103.0")])
def test_stop_tick_rounding(side, expected):
    p = dict(entry_px=100, width=.03005, side=side)
    assert risk.stop_price(p, {"tickSz": ".1"}) == expected


def test_five_slots_never_exceed_cash_even_with_coarse_lots():
    specs = {str(i): {"ctVal": "1", "ctValCcy": "USDT", "lotSz": "1", "minSz": "1"} for i in range(5)}
    balance = [{"details": [{"ccy": "USDT", "cashBal": "11", "availBal": "11"}]}]
    s = state()
    sizes = []
    for i in range(5):
        equity, budget = live.entry_budget(balance, s, specs, {})
        size = live.size_for_signal(None, {"inst_id": str(i)}, 1, equity, specs, budget=budget)
        sizes.append(Decimal(size))
        s["positions"][str(i)] = dict(inst_id=str(i), size=size, entry_px=1)
    assert sum(sizes) == 10 and sum(sizes) <= Decimal("11") * (1 - Decimal(str(live.CAPITAL_BUFFER)))


def test_pending_orders_and_exchange_available_balance_both_cap_budget():
    specs = {"A": {"ctVal": "1", "ctValCcy": "USDT"}}
    s = state()
    s["pending_entries"] = {"id": dict(inst_id="A", size="80", entry_px=1)}
    b = [{"totalEq": "10000", "details": [{"ccy": "USDT", "cashBal": "100", "availBal": "10"}]}]
    equity, budget = live.entry_budget(b, s, specs, {})
    assert equity == 100 and budget == 9
    assert live.entry_budget([{"totalEq": "10000"}], s, specs, {}) == (0, 0)


def test_stale_symbol_does_not_prevent_fresh_symbol_entry(loop, monkeypatch):
    client, now = loop
    sigs = {"OLD": signal("OLD", now=now - pd.Timedelta(hours=1)), "NEW": signal("NEW", now=now)}
    live.run_once(client, state(), True, prepared_signals=sigs)
    assert [x[0] for x in client.orders] == ["NEW"]


def test_stale_metrics_prevents_entry_even_with_fresh_bar(loop):
    client, now = loop
    sig = signal("BTCUSDT", "crypto", now)
    sig["metrics_ts"] = int((now - pd.Timedelta(hours=1)).timestamp() * 1000)
    live.run_once(client, state(), True, prepared_signals={"BTCUSDT": sig})
    assert not client.orders


@pytest.mark.parametrize("top,sub,expected", [("0", "51008", True), ("1", "51008", True),
                                               ("0", "50004", False)])
def test_client_validates_per_order_status(top, sub, expected):
    c = object.__new__(api.DemoClient)
    c.key = c.secret = c.passphrase = "test"
    response = Mock(status_code=200)
    response.json.return_value = {"code": top, "data": [{"sCode": sub, "sMsg": "test"}]}
    c.s = Mock()
    c.s.request.return_value = response
    with pytest.raises(api.OKXAPIError) as exc:
        c.order("TEST", "buy", "1", exp_time=123)
    assert exc.value.rejected is expected
    assert c.s.request.call_args.kwargs["headers"]["expTime"] == "123"


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "DB_PATH", tmp_path / "ledger.sqlite3")
    monkeypatch.setattr(sync_loop.time, "sleep", lambda _: None)


def fill(i, inst="A"):
    return dict(billId=str(i), tradeId=str(i), instId=inst, ts="1790000000000",
                fillTime="1790000000000", fillSz="1", fee="-.1", feeCcy="USDT")


def test_backfill_resumes_after_failure_without_losing_page(ledger):
    c = Mock()
    c.history_page.side_effect = [[fill(i) for i in range(250, 150, -1)], requests.Timeout()]
    with pytest.raises(requests.Timeout):
        sync_loop.sync_stream(c, "fills")
    with api.db_connect() as db:
        assert db.execute("SELECT count(*) FROM fills").fetchone()[0] == 100
        checkpoint = json.loads(db.execute("SELECT raw_json FROM account_sync_state").fetchone()[0])
    assert checkpoint["after"] == "151" and checkpoint["active"]
    c.history_page.side_effect = [[fill(i) for i in range(150, 50, -1)], [fill(i) for i in range(50, 0, -1)], []]
    count, complete = sync_loop.sync_stream(c, "fills")
    assert count == 150 and complete
    with api.db_connect() as db:
        assert db.execute("SELECT count(*) FROM fills").fetchone()[0] == 250


def test_trade_ids_are_scoped_by_instrument_and_replay_is_idempotent(ledger):
    api.save_account_data([], [fill(1, "A"), fill(1, "B")], [])
    api.save_account_data([], [fill(1, "A"), fill(1, "B")], [])
    with api.db_connect() as db:
        assert db.execute("SELECT count(*) FROM fills").fetchone()[0] == 2
        assert db.execute("SELECT ts FROM fills LIMIT 1").fetchone()[0].startswith("2026-")


def test_failed_fill_stream_does_not_block_bill_sync(ledger):
    c = Mock()
    def page(stream, **params):
        if stream == "fills":
            raise requests.Timeout()
        return [] if params.get("after") else [dict(billId="1", ts="1790000000000", type="8", subType="174", balChg="1")]
    c.history_page.side_effect = page
    c.balance.return_value = []
    c.positions.return_value = []
    sync_loop.sync_once(c)
    with api.db_connect() as db:
        assert db.execute("SELECT count(*) FROM bills").fetchone()[0] == 1


def test_page_limit_leaves_durable_active_cursor(ledger):
    c = Mock()
    c.history_page.return_value = [fill(100)]
    assert sync_loop.sync_stream(c, "fills", max_pages=1) == (1, False)
    with api.db_connect() as db:
        p = json.loads(db.execute("SELECT raw_json FROM account_sync_state").fetchone()[0])
    assert p["active"] and p["after"] == "100" and "watermark" not in p


@pytest.mark.parametrize("stale_part", ["klines", "metrics", "features", "oi_source"])
def test_crypto_builder_rejects_only_the_stale_symbol(monkeypatch, stale_part):
    stamp = int(time.time() * 1000) // live.BAR_MS * live.BAR_MS - live.BAR_MS
    class Data:
        def klines(self, sym):
            ts = stamp - (7200_000 if sym == "ETHUSDT" and stale_part == "klines" else 0)
            return pd.DataFrame({"ts": [ts], "close": [100.]})

        def metrics(self, sym):
            ts = stamp - live.BAR_MS
            if sym == "ETHUSDT" and stale_part == "metrics":
                ts -= 7200_000
            source = ts - (7200_000 if sym == "ETHUSDT" and stale_part == "oi_source" else 0)
            return pd.DataFrame({"ts": [ts, ts + live.BAR_MS],
                                 "oi_source_ts": [source, source + live.BAR_MS],
                                 "ratio_source_ts": [ts, ts + live.BAR_MS]})

    def frame(sym, *args, **kwargs):
        ts = stamp - (7200_000 if sym == "ETHUSDT" and stale_part == "features" else 0)
        return pd.DataFrame({"ts": [ts], "sigma": [.01], "ret_96": [0.],
                             "dist_ema_96": [0.], "dist_ema_384": [0.], "width_c": [.03]})
    model = Mock()
    model.feature_name.return_value = ["sigma", "sym_code"]
    model.predict.return_value = [.9]
    monkeypatch.setattr(live, "SYMBOLS", ("BTCUSDT", "ETHUSDT", "SOLUSDT"))
    monkeypatch.setattr(live.build, "build_symbol", frame)
    monkeypatch.setattr(live.lgb, "Booster", lambda **kwargs: model)
    monkeypatch.setattr(live.pd, "read_csv", lambda *a, **k: pd.DataFrame({"fold": [1], "hi_0.01": [.8], "lo_0.01": [.2]}))
    assert set(live.build_crypto_signals(Data())) == {"BTCUSDT", "SOLUSDT"}


def test_unconfirmed_candles_are_not_used():
    data = live.LiveData(None)
    confirmed = ["1000", "1", "1", "1", "1", "1", "1", "1", "1"]
    unfinished = ["2000", "2", "2", "2", "2", "2", "2", "2", "0"]
    data._okx = Mock(return_value={"code": "0", "data": [confirmed, unfinished]})
    assert data.klines("BTCUSDT").ts.tolist() == [1000]


def test_stats_uses_fills_and_both_funding_directions(ledger, capsys):
    f = fill(1)
    f["fillPnl"] = "2"
    api.save_account_data([], [f], [dict(billId="a", ts="1790000000000", type="8", subType="173", ccy="USDT", balChg="-.3"),
                                   dict(billId="b", ts="1790000000000", type="8", subType="174", ccy="USDT", balChg=".2")])
    api.print_stats()
    output = capsys.readouterr().out
    assert "realized_pnl=2.00000000" in output and "total=1.80000000" in output


def test_legacy_fill_identity_migration_is_idempotent(ledger):
    f = fill(1)
    with api.db_connect() as db:
        db.execute("INSERT INTO fills(trade_id,ts,inst_id,raw_json) VALUES (?,?,?,?)", ("1", "old", "A", json.dumps(f)))
    api.save_account_data([], [f], [])
    with api.db_connect() as db:
        rows = db.execute("SELECT trade_id,ts FROM fills").fetchall()
    assert len(rows) == 1 and rows[0][0] == "A:1" and rows[0][1] != "old"


def test_existing_exchange_order_blocks_reentry(loop):
    client, now = loop
    client.pending_orders = Mock(return_value=[{"ordId": "old-stop-child", "reduceOnly": "true"}])
    live.run_once(client, state(), True, prepared_signals={"TEST": signal("TEST", now=now)})
    assert not client.orders


def test_rejected_stop_leaves_no_orphan_identity(loop):
    client, now = loop
    s = state({"p": position(now=now)})
    client.stop_order = Mock(side_effect=api.OKXAPIError("51333", "invalid algo parameters"))
    risk.protect_positions(client, s, {"TEST-USDT-SWAP": {"tickSz": ".01"}}, lambda _: None)
    assert s["positions"]["p"]["force_exit"]
    assert not s["protective_orders"] and "stop_client_id" not in s["positions"]["p"]


def test_partial_entry_is_adopted_and_remainder_canceled(loop):
    client, now = loop
    p = position(now=now)
    s = state()
    s["pending_entries"] = {"entry": p}
    client.order_detail = Mock(return_value={"ordId": "o1", "state": "partially_filled", "accFillSz": ".4", "avgPx": "101"})
    client.cancel_order = Mock()
    live.reconcile_pending_entries(client, s)
    assert s["positions"][p["inst_id"] + "|long"]["size"] == "0.4"
    client.cancel_order.assert_called_once_with(p["inst_id"], "o1")
    assert "entry" in s["pending_entries"]
    client.order_detail.return_value["state"] = "canceled"
    live.reconcile_pending_entries(client, s)
    assert not s["pending_entries"] and s["positions"] and s["trades"] == 1
