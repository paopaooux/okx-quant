from __future__ import annotations

import pandas as pd

from strategies.stocks.config import Config
from strategies.stocks.research.news_strategy import build_events
from strategies.stocks.research.backtest import Rules, run
from strategies.stocks.research.stock_categories import classify_universe, validate_categories


def test_stock_config_uses_external_archive(monkeypatch):
    monkeypatch.setenv("STOCK_ALPHA_ROOT", "/tmp/stock-alpha-archive")
    cfg = Config()
    assert cfg.data_dir == "/tmp/stock-alpha-archive/data"
    assert cfg.result_dir == "/tmp/stock-alpha-archive/results"


def test_empty_filings_produce_no_events():
    frames = {"XTEST-USDT": pd.DataFrame()}
    filings = pd.DataFrame(columns=["form", "accepted", "instId"])
    assert build_events(filings, frames).empty


def test_event_entry_is_first_completed_bar_after_decision():
    index = pd.date_range(
        "2026-07-16 20:00:00", periods=8, freq="5min", tz="UTC"
    )
    frame = pd.DataFrame({"close": [100, 100, 101, 101, 101, 101, 101, 101]}, index=index)
    filings = pd.DataFrame([{
        "form": "8-K",
        "accepted": "2026-07-16T20:00:00Z",
        "instId": "XTEST-USDT",
        "ticker": "TEST",
        "items": "8.01",
    }])

    decision = pd.Timestamp("2026-07-16 20:31:00", tz="UTC")
    events = build_events(filings, {"XTEST-USDT": frame}, observe_minutes=31)

    assert len(events) == 1
    assert events.iloc[0].event_ts >= decision
    assert events.iloc[0].event_ts == pd.Timestamp("2026-07-16 20:35:00", tz="UTC")


def test_stock_categories_cover_current_universe():
    universe = pd.read_csv("data/stocks_swap/universe.csv")
    validate_categories(universe)
    categorized = classify_universe(universe)
    assert categorized.category.notna().all()
    assert categorized.category_size.min() >= 1


def test_backtest_direction_is_explicit():
    assert Rules(direction="both").direction == "both"
    assert Rules(direction="long").direction == "long"
    assert Rules(direction="short").direction == "short"


def test_backtest_direction_filters_sides():
    index = pd.date_range("2026-07-16 00:00:00", periods=8, freq="5min", tz="UTC")
    frame = pd.DataFrame({
        "open": [100.0] * 8,
        "high": [101.0] * 8,
        "low": [99.0] * 8,
        "close": [100.0] * 8,
        "volume_quote": [1_000.0] * 8,
    }, index=index)
    events = pd.DataFrame([
        {"inst_id": "XTEST-USDT", "event_ts": index[1], "event_type": "test",
         "side": 1, "resolve_ts": index[3], "event_day": index[1].date(), "deviation": 0.1},
        {"inst_id": "XTEST-USDT", "event_ts": index[2], "event_type": "test",
         "side": -1, "resolve_ts": index[4], "event_day": index[2].date(), "deviation": -0.1},
    ])
    cfg = Config(fee_bps=0.0, slippage_bps=0.0)
    common = dict(horizon="to_open", resolve_offset_minutes=0, stop_loss_bps=0,
                  max_concurrent=2, max_per_day=10, rank_column="abs_deviation")
    long_trades = run(events, {"XTEST-USDT": frame}, cfg, Rules(**common, direction="long")).trades
    short_trades = run(events, {"XTEST-USDT": frame}, cfg, Rules(**common, direction="short")).trades
    assert long_trades.side.tolist() == [1]
    assert short_trades.side.tolist() == [-1]
