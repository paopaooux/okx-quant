from __future__ import annotations

import pandas as pd
import pytest

from strategies.stocks.config import Config
from strategies.stocks.research.backtest import Rules, run, summarize_trades
from strategies.stocks.research.stock_categories import classify_universe, validate_categories


def test_stock_config_uses_external_archive(monkeypatch):
    monkeypatch.setenv("STOCK_ALPHA_ROOT", "/tmp/stock-alpha-archive")
    cfg = Config()
    assert cfg.data_dir == "/tmp/stock-alpha-archive/data"
    assert cfg.result_dir == "/tmp/stock-alpha-archive/results"


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


def test_trade_summary_reports_significance_and_direction_contributions():
    ts = pd.date_range("2026-01-01", periods=4, freq="D", tz="UTC")
    trades = pd.DataFrame({
        "net": [0.10, 0.20, -0.05, 0.15],
        "pnl": [0.10, 0.20, -0.05, 0.15],
        "side": [1, 1, -1, -1],
        "entry_ts": ts,
        "exit_ts": ts,
        "hold_hours": [1.0] * 4,
        "reason": ["deadline"] * 4,
    })
    summary = summarize_trades(trades, Rules())
    assert summary["sqn"] > 0
    assert 0 <= summary["mean_profit_pvalue"] <= 1
    assert summary["long_profit_pct"] == pytest.approx(30.0)
    assert summary["short_profit_pct"] == pytest.approx(10.0)
