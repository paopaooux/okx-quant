from __future__ import annotations

import pandas as pd

from strategies.stocks.config import Config
from strategies.stocks.research.news_strategy import build_events


def test_stock_config_uses_external_archive(monkeypatch):
    monkeypatch.setenv("STOCK_ALPHA_ROOT", "/tmp/stock-alpha-archive")
    cfg = Config()
    assert cfg.data_dir == "/tmp/stock-alpha-archive/data"
    assert cfg.result_dir == "/tmp/stock-alpha-archive/results"


def test_empty_filings_produce_no_events():
    frames = {"XTEST-USDT": pd.DataFrame()}
    filings = pd.DataFrame(columns=["form", "accepted", "instId"])
    assert build_events(filings, frames).empty
