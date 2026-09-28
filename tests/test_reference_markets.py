import pandas as pd
import pytest

from scripts.live import auto_demo as live
from scripts.live.combination_policy import entry_metadata, entry_rejection, stock_entry_time_valid
from strategies.stocks.config import Config
from strategies.stocks.market import reference
from test_live_combination_policy import signal, state, loop


@pytest.mark.parametrize("inst,market", [
    ("KORU-USDT-SWAP", "XNYS"), ("SKHY-USDT-SWAP", "XNYS"),
    ("SONY-USDT-SWAP", "XNYS"), ("CSOPSKHYNIX2L-USDT-SWAP", "XHKG"),
    ("MINIMAX-USDT-SWAP", "XHKG"), ("KIOXIA-USDT-SWAP", "XTKS"),
    ("SKHYNIX-USDT-SWAP", "XKRX"), ("ANTHROPIC-USDT-SWAP", None),
    ("UNREVIEWED-USDT-SWAP", None),
])
def test_listing_market_not_company_nationality(inst, market):
    assert reference.reference_market(inst) == market


def test_local_hynix_listing_shares_underlying_limit():
    from test_live_combination_policy import position, NOW
    s = state({"a": position("SKUU-USDT-SWAP")})
    assert entry_rejection(s, signal("SKHYNIX-USDT-SWAP"), NOW) == "same_underlying"


def test_hong_kong_holiday_and_japan_long_holiday():
    hk = reference.closed_windows("XHKG", pd.Timestamp("2026-09-30T00:00Z"), pd.Timestamp("2026-10-02T12:00Z"))
    assert pd.Timestamp("2026-10-01T01:30Z") not in set(hk.open_ts)
    jp = reference.closed_windows("XTKS", pd.Timestamp("2026-09-18T00:00Z"), pd.Timestamp("2026-09-24T12:00Z"))
    weekend = jp[jp.open_ts.eq(pd.Timestamp("2026-09-24T00:00Z"))].iloc[0]
    assert weekend.close_ts == pd.Timestamp("2026-09-18T06:30Z")


def test_us_dst_changes_utc_open():
    windows = reference.closed_windows("XNYS", pd.Timestamp("2026-03-05T00:00Z"), pd.Timestamp("2026-03-10T00:00Z"))
    assert pd.Timestamp("2026-03-06T14:30Z") in set(windows.open_ts)
    assert pd.Timestamp("2026-03-09T13:30Z") in set(windows.open_ts)


def test_lunch_not_a_new_overnight_event_and_timestamp_units():
    index = pd.DatetimeIndex(["2026-09-16T02:00Z", "2026-09-16T04:30Z", "2026-09-16T05:30Z"])
    assert reference.market_state(index.as_unit("us"), "XHKG").tolist() == ["cash", "closed", "cash"]
    windows = reference.closed_windows("XHKG", index[0], index[-1])
    assert not windows.open_ts.eq(pd.Timestamp("2026-09-16T05:00Z")).any()


def test_signal_resolves_to_local_open_and_not_us_open():
    index = pd.date_range("2026-09-15T08:00Z", "2026-09-16T00:00Z", freq="5min")
    frame = pd.DataFrame({"close": 100., "volume_quote": 1000.}, index=index)
    frame.loc[index >= pd.Timestamp("2026-09-15T20:00Z"), "close"] = 107.
    found = reference.stock_events({"MINIMAX-USDT-SWAP": frame}, Config(dislocation_bps=600), now=index[-1])
    row = found.iloc[0]
    assert row.resolve_ts == pd.Timestamp("2026-09-16T01:30Z")
    assert row.reference_market == "XHKG"
    sig = signal("MINIMAX-USDT-SWAP", now=row.event_ts)
    sig.update(resolve_ts=row.resolve_ts.isoformat(), reference_market=row.reference_market,
               calendar_policy_version=row.calendar_policy_version)
    assert pd.Timestamp(entry_metadata(sig)["deadline_ts"]) == pd.Timestamp("2026-09-16T02:30Z")


def test_unknown_market_and_open_cash_session_produce_no_signal():
    index = pd.date_range("2026-09-15T08:00Z", "2026-09-16T02:00Z", freq="5min")
    frame = pd.DataFrame({"close": 107., "volume_quote": 1000.}, index=index)
    assert reference.stock_events({"ANTHROPIC-USDT-SWAP": frame}, Config()).empty
    assert reference.stock_events({"MINIMAX-USDT-SWAP": frame}, Config(), now=index[-1]).empty


def test_cash_open_blocks_entry_even_before_holding_deadline():
    now = pd.Timestamp("2026-09-16T01:30Z")
    sig = signal("MINIMAX-USDT-SWAP", now=now-pd.Timedelta(minutes=5))
    sig["resolve_ts"] = now.isoformat()
    assert not stock_entry_time_valid(sig, now)
    assert entry_rejection(state(), sig, now) == "cash_session_started_or_unknown"


def test_cached_signal_expires_at_reference_open(tmp_path, monkeypatch):
    import json
    now = pd.Timestamp("2026-09-16T01:30Z")
    status = tmp_path / "status.json"
    status.write_text(json.dumps({"updated_at": now.isoformat(), "candles": {"last_success_at": now.isoformat()}}))
    sig = signal("MINIMAX-USDT-SWAP", now=now-pd.Timedelta(minutes=1))
    sig["resolve_ts"] = now.isoformat()
    monkeypatch.setattr(live, "STOCK_STATUS", status)
    monkeypatch.setattr(live, "_stock_signal_cache_key", (now.isoformat(), reference.VERSION))
    monkeypatch.setattr(live, "_stock_signal_cache", {"MINIMAX": sig})
    assert live.build_stock_signals(now) == {}


def test_calendar_out_of_coverage_never_falls_back_to_us():
    with pytest.raises(ValueError, match="coverage"):
        reference.closed_windows("XHKG", pd.Timestamp("1900-01-01T00:00Z"), pd.Timestamp("1900-01-02T00:00Z"))


def test_market_open_during_account_checks_blocks_submission(loop, monkeypatch):
    client, now = loop
    sig = signal(now=now)
    s = state()
    monkeypatch.setattr(live, "stock_entry_time_valid", lambda *args: False)
    live.run_once(client, s, True, prepared_signals={sig["inst_id"]: sig})
    assert not client.orders and not s["pending_entries"] and not s["stock_entries_by_day"]


def test_existing_position_keeps_old_deadline():
    from scripts.live.combination_policy import migrate_policy_state
    from test_live_combination_policy import position
    now = pd.Timestamp("2026-09-16T02:00Z")
    p = position("MINIMAX-USDT-SWAP", now=now)
    old = "2026-09-16T14:30:00+00:00"
    p["deadline_ts"] = old
    s = state({"MINIMAX": p})
    migrate_policy_state(s, now)
    assert p["deadline_ts"] == old


def test_canonical_backtest_defaults_to_reference_calendar(monkeypatch):
    from scripts.stocks.research_offhours_candidates import _args
    monkeypatch.setattr("sys.argv", ["research_offhours_candidates"])
    assert _args().calendar_mode == "reference"
    monkeypatch.setattr("sys.argv", ["research_offhours_candidates", "--calendar-mode", "legacy-us"])
    assert _args().calendar_mode == "legacy-us"
