import copy
import json

import pandas as pd
import pytest

from scripts.live import auto_demo as live
from scripts.live import stock_data_loop
from scripts.live.combination_policy import (
    POLICY, entry_metadata, entry_rejection, exit_reason, migrate_policy_state,
    stock_deadline, stock_instruments,
)


NOW = pd.Timestamp("2026-09-07T10:00:00Z")


def signal(inst="TEST-USDT-SWAP", asset="stock", now=NOW):
    return dict(inst_id=inst, symbol=inst, asset_type=asset, strategy="xstock_hybrid",
                side="long", event_ts=now.isoformat(), resolve_ts=(now + pd.Timedelta(hours=2)).isoformat(),
                bar=int(now.timestamp() * 1000), close=100., width=.03,
                observe_move_bps=600., p_up=None)


def position(inst="TEST-USDT-SWAP", now=NOW, legacy=False):
    sig = signal(inst, now=now)
    pos = {**sig, **entry_metadata(sig), "entry_px": 100., "size": "1"}
    if legacy:
        for k in ("policy_version", "deadline_ts", "take_width"):
            pos.pop(k)
        pos["width"] = .06
    return pos


def state(positions=None):
    return dict(policy_version=POLICY.version, positions=positions or {}, last_bars={},
                stock_entries_by_day={}, pending_entries={}, trades=0)


def test_frozen_rules_match_original_metadata():
    meta = json.loads(open("results/stocks_offhours_research/metadata.json").read())
    assert POLICY.stock_trigger_bps == meta["dislocation_bps"]
    assert POLICY.stock_stop_bps == meta["stop_loss_bps"]
    assert POLICY.stock_resolve_minutes == meta["resolve_offset_minutes"]
    assert POLICY.stock_slots == meta["slots"]
    assert POLICY.stock_entries_per_utc_day == meta["max_per_day"]
    assert POLICY.stock_pool == meta["pool"]
    assert POLICY.stock_take_bps == 0
    assert POLICY.shared_slots == 5 and POLICY.slot_weight == .2


def test_all_stock_instruments_no_tech_or_leveraged_exclusion():
    universe = pd.DataFrame({"instId": ["A-USDT-SWAP", "B-USDT-SWAP"],
                             "ticker": ["NOT_TECH", "ALSO_NOT_TECH"], "kind": ["single", "leveraged"]})
    assert stock_instruments(universe) == ["A-USDT-SWAP", "B-USDT-SWAP"]
    assert stock_data_loop.stock_instruments(universe) == stock_instruments(universe)


def test_stock_deadline_open_plus_hour_and_30h_cap():
    assert stock_deadline(NOW, NOW + pd.Timedelta(hours=2)) == NOW + pd.Timedelta(hours=3)
    assert stock_deadline(NOW, NOW + pd.Timedelta(days=3)) == NOW + pd.Timedelta(hours=30)


@pytest.mark.parametrize("side,mark", [("long", 97.), ("short", 103.)])
def test_new_stock_stop_and_no_fixed_take(side, mark):
    pos = position()
    pos["side"] = side
    assert exit_reason(pos, mark, NOW) == "stop"
    assert exit_reason(pos, 110. if side == "long" else 90., NOW) is None
    assert exit_reason(pos, 100., NOW + pd.Timedelta(hours=3)) == "deadline"


def test_crypto_timeout_uses_next_open_clock():
    meta = entry_metadata(signal(asset="crypto"))
    assert pd.Timestamp(meta["deadline_ts"]) == NOW + pd.Timedelta(minutes=49 * 15)
    assert meta["take_width"] == .03


def test_legacy_positions_not_tightened_and_migration_day_capped():
    pos = position(legacy=True)
    s = dict(positions={"legacy": pos}, trades=27)
    migrate_policy_state(s, NOW)
    assert pos["policy_version"] == "legacy_preserved"
    assert pos["width"] == pos["take_width"] == .06
    assert pd.Timestamp(pos["deadline_ts"]) == NOW + pd.Timedelta(hours=30)
    assert exit_reason(pos, 96., NOW) is None
    assert exit_reason(pos, 106., NOW) == "target"
    assert entry_rejection(s, signal("NEW"), NOW) == "stock_daily_limit"
    assert entry_rejection(s, signal("NEW"), NOW + pd.Timedelta(days=1)) is None


def test_stock_cap_and_shared_pool_count_pending_orders():
    s = state({str(i): position(str(i)) for i in range(3)})
    assert entry_rejection(s, signal("NEW"), NOW) == "stock_capacity"
    assert entry_rejection(s, signal("BTC", "crypto"), NOW) is None
    s["pending_entries"] = {"x": signal("BTC", "crypto"), "y": signal("ETH", "crypto")}
    assert entry_rejection(s, signal("ADA", "crypto"), NOW) == "shared_capacity"


def test_daily_counter_survives_json_restart_and_resets_utc_day():
    s = state()
    s["stock_entries_by_day"]["2026-09-07"] = 2
    s = json.loads(json.dumps(s))
    assert entry_rejection(s, signal(), NOW) == "stock_daily_limit"
    assert entry_rejection(s, signal(), pd.Timestamp("2026-09-08T00:00:00Z")) is None


def test_management_signal_never_reopens():
    sig = signal()
    sig["management_only"] = True
    assert entry_rejection(state(), sig, NOW) == "management_only_or_flat"


def test_remote_api_failure_is_not_empty_account():
    class Client:
        def positions(self, *args):
            raise RuntimeError("network error")
    with pytest.raises(RuntimeError, match="network error"):
        live.remote_positions(Client())


def test_invalid_state_fails_closed(tmp_path, monkeypatch):
    p = tmp_path / "state.json"
    p.write_text("{")
    monkeypatch.setattr(live, "STATE_PATH", p)
    with pytest.raises(RuntimeError, match="refusing to forget"):
        live.load_state()


class FakeClient:
    s = None

    def __init__(self):
        self.orders = []
        self.detail_state = "filled"

    def order(self, inst, side, size, *args, **kwargs):
        self.orders.append((inst, side, size, args, kwargs))
        return [{"ordId": "123"}]

    def order_detail(self, *args, **kwargs):
        return dict(state=self.detail_state, avgPx="100", accFillSz="1")

    def balance(self):
        return [{"totalEq": "1000"}]


@pytest.fixture
def loop(monkeypatch):
    now = pd.Timestamp.now(tz="UTC")
    class FakeData:
        def __init__(self, *args):
            pass

        def ticker(self, *args):
            return 100., int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
    monkeypatch.setattr(live, "LiveData", FakeData)
    monkeypatch.setattr(live, "build_crypto_signals", lambda data: {})
    monkeypatch.setattr(live, "build_stock_signals", lambda now: {})
    monkeypatch.setattr(live, "save_state", lambda s: None)
    monkeypatch.setattr(live, "save_balance", lambda rows: None)
    monkeypatch.setattr(live, "save_strategy_snapshots", lambda rows: None)
    monkeypatch.setattr(live, "contract_specs", lambda client: {})
    monkeypatch.setattr(live, "size_for_signal", lambda *args: "1")
    monkeypatch.setattr(live, "remote_positions", lambda client: {})
    return FakeClient(), now


def test_closed_stock_does_not_reopen_from_synthetic_signal(loop, monkeypatch):
    client, now = loop
    pos = position(now=now - pd.Timedelta(hours=31), legacy=True)
    s = state({"TEST-USDT-SWAP|long": pos})
    remote = {"TEST-USDT-SWAP|long": {**pos, "upl": 0.}}
    monkeypatch.setattr(live, "remote_positions", lambda client: remote)
    live.run_once(client, s, True)
    assert not s["positions"]
    assert len(client.orders) == 1
    assert client.orders[0][3][1] is True


def test_exits_run_when_crypto_model_fails(loop, monkeypatch):
    client, now = loop
    pos = position(now=now - pd.Timedelta(hours=4))
    s = state({"TEST-USDT-SWAP|long": pos})
    monkeypatch.setattr(live, "remote_positions", lambda client: {"TEST-USDT-SWAP|long": {**pos, "upl": 0.}})
    def broken(data):
        raise RuntimeError("no features")
    monkeypatch.setattr(live, "build_crypto_signals", broken)
    live.run_once(client, s, True)
    assert len(client.orders) == 1 and not s["positions"]


def test_new_entry_persists_deadline_and_daily_limit(loop, monkeypatch):
    client, now = loop
    signals = {str(i): signal(str(i), now=now) for i in range(3)}
    monkeypatch.setattr(live, "build_stock_signals", lambda now: signals)
    s = state()
    live.run_once(client, s, True)
    assert len(client.orders) == len(s["positions"]) == 2
    assert s["stock_entries_by_day"][str(now.date())] == 2
    for pos in s["positions"].values():
        assert pos["width"] == .03 and pos["take_width"] == 0
        assert pos["resolve_ts"] and pos["deadline_ts"] and pos["policy_version"] == POLICY.version


def test_uncertain_entry_durable_and_not_resubmitted(loop, monkeypatch):
    client, now = loop
    monkeypatch.setattr(live, "build_stock_signals", lambda now_: {"TEST-USDT-SWAP": signal(now=now)})
    snapshots = []
    monkeypatch.setattr(live, "save_state", lambda s: snapshots.append(copy.deepcopy(s)))
    def broken_order(*args, **kwargs):
        raise RuntimeError("timeout")
    client.order = broken_order
    s = state()
    with pytest.raises(RuntimeError, match="timeout"):
        live.run_once(client, s, True)
    assert len(snapshots[-1]["pending_entries"]) == 1
    assert snapshots[-1]["stock_entries_by_day"][str(now.date())] == 1
    assert entry_rejection(s, signal(now=now), now) == "same_instrument"


def test_config_drift_is_rejected(monkeypatch):
    live.validate_combination_config()
    monkeypatch.setattr(live, "STOCK_STOP_BPS", 600.)
    with pytest.raises(RuntimeError, match="differs"):
        live.validate_combination_config()


def test_stock_signal_uses_600bp_all_pool_and_holiday_open(tmp_path, monkeypatch):
    universe = pd.DataFrame({"instId": ["TEST-USDT-SWAP"], "ticker": ["NOT_TECH"]})
    universe.to_csv(tmp_path / "universe.csv", index=False)
    status = tmp_path / "status.json"
    status.write_text(json.dumps({"updated_at": NOW.isoformat(),
                                 "candles": {"last_success_at": NOW.isoformat()}}))
    monkeypatch.setattr(live, "STOCK_DATA", tmp_path)
    monkeypatch.setattr(live, "STOCK_STATUS", status)
    monkeypatch.setattr(live, "_stock_signal_cache_key", None)
    idx = pd.date_range(NOW - pd.Timedelta(minutes=5), periods=4, freq="5min")
    frame = pd.DataFrame({"close": [100., 106., 200., 300.]}, index=idx)
    monkeypatch.setattr(live.stock_data, "load_panel", lambda ids, *args: {ids[0]: frame})
    monkeypatch.setattr(live.stock_data, "to_bar_end", lambda frames, bar: frames)
    def detect(frames, cfg, windows):
        assert cfg.dislocation_bps == 600
        assert frames["TEST-USDT-SWAP"].index.max() == NOW
        assert windows.open_ts.iloc[0] == pd.Timestamp("2026-09-08T13:30:00Z")
        return pd.DataFrame([dict(inst_id="TEST-USDT-SWAP", event_ts=NOW, side=-1,
                                  resolve_ts=windows.open_ts.iloc[0], deviation=.06)])
    monkeypatch.setattr(live.stock_events, "off_hours_dislocation", detect)
    sig = live.build_stock_signals(NOW)["TEST-USDT-SWAP"]
    assert sig["side"] == "short" and sig["width"] == .03
    assert sig["close"] == 106 and sig["ticker"] == "NOT_TECH"


def test_client_order_id_is_sent_and_lookup_supported():
    from scripts.live.okx_demo import DemoClient
    client = object.__new__(DemoClient)
    calls = []
    def request(method, path, **kwargs):
        calls.append((method, path, kwargs))
        return [{"state": "filled"}]
    client._request = request
    client.order("TEST-USDT-SWAP", "buy", "1", client_order_id="unique123")
    assert calls[-1][2]["body"]["clOrdId"] == "unique123"
    client.order_detail("TEST-USDT-SWAP", client_order_id="unique123")
    assert calls[-1][2]["params"] == {"instId": "TEST-USDT-SWAP", "clOrdId": "unique123"}


def test_original_299_trade_schedule_obeys_live_admission_rules():
    trades = pd.read_csv("results/combinations/latest/combined_trades.csv")
    for col in ("entry_ts", "exit_ts"):
        trades[col] = pd.to_datetime(trades[col], utc=True)
    s = state()
    for row in trades.sort_values(["entry_ts", "signal_strength", "strategy"], ascending=[True, False, True]).itertuples():
        now = row.entry_ts
        s["positions"] = {k: p for k, p in s["positions"].items() if p["exit_ts"] > now}
        asset = "stock" if row.strategy == "xstock_hybrid" else "crypto"
        sig = signal(row.symbol, asset, now)
        sig["side"] = "long" if row.side > 0 else "short"
        assert entry_rejection(s, sig, now) is None
        s["positions"][row.symbol] = {**sig, "exit_ts": row.exit_ts}
        if asset == "stock":
            day = str(now.date())
            s["stock_entries_by_day"][day] = s["stock_entries_by_day"].get(day, 0) + 1
    assert len(trades) == 299


def test_uncertain_exit_does_not_block_other_positions(loop, monkeypatch):
    client, now = loop
    a, b = position("A", now - pd.Timedelta(hours=4)), position("B", now - pd.Timedelta(hours=4))
    a["exit_client_id"] = "unknown"
    s = state({"A": a, "B": b})
    original_detail = client.order_detail
    def detail(inst, *args, **kwargs):
        if inst == "A":
            raise RuntimeError("not found yet")
        return original_detail(inst, *args, **kwargs)
    client.order_detail = detail
    live.manage_positions(client, s, live.LiveData(None), now, True)
    assert "A" in s["positions"] and "B" not in s["positions"]
    assert s["entry_reconciliation_ok"] is False
