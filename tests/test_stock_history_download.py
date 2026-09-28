import pandas as pd
import pytest

from scripts.analysis import download_stock_history as downloader


@pytest.fixture
def archive(tmp_path, monkeypatch):
    tmp_path.joinpath("5m").mkdir()
    monkeypatch.setattr(downloader.time, "sleep", lambda _: None)
    start = pd.Timestamp("2026-03-01T00:00Z")
    stamps = pd.date_range(start, periods=4, freq="5min")
    batches = [[str(int(t.timestamp()*1000)), "100", "101", "99", "100", "1", "1", "100", "1"]
               for t in reversed(stamps)]
    calls = []

    def get(self, endpoint, params):
        calls.append(params)
        return [r for r in batches if int(r[0]) < int(params["after"])][:2]

    monkeypatch.setattr(downloader.OKXClient, "_get", get)
    return tmp_path, start, batches, calls


def test_download_pagination_and_resume(archive):
    root, start, _, calls = archive
    end = start+pd.Timedelta(minutes=20)
    row = downloader.download("TEST", start, root, start, end, None)
    assert row["rows"] == 4 and row["missing_bars"] == 0
    assert len(calls) == 2
    row2 = downloader.download("TEST", start, root, start, end, None)
    assert len(calls) == 2
    assert row2["sha256"] == row["sha256"]


def test_later_listing_does_not_create_fake_missing_history(archive):
    root, start, batches, _ = archive
    del batches[2:]
    row = downloader.download("TEST", start+pd.Timedelta(minutes=10), root, start,
                              start+pd.Timedelta(minutes=20), None)
    assert row["missing_bars"] == 0 and row["rows"] == 2


def test_missing_candle_is_reported(archive):
    root, start, batches, _ = archive
    del batches[1]
    row = downloader.download("TEST", start, root, start, start+pd.Timedelta(minutes=20), None)
    assert row["missing_bars"] == 1


def test_unconfirmed_candle_is_not_used(archive):
    root, start, batches, _ = archive
    batches[1][-1] = "0"
    row = downloader.download("TEST", start, root, start, start+pd.Timedelta(minutes=20), None)
    assert row["missing_bars"] == 1 and row["rows"] == 3


def test_non_progressing_page_fails(archive, monkeypatch):
    root, start, batches, _ = archive
    monkeypatch.setattr(downloader.OKXClient, "_get", lambda *args: batches[:1])
    with pytest.raises(ValueError, match="Non-progressing"):
        downloader.download("TEST", start, root, start, start+pd.Timedelta(minutes=20), None)


def test_minute_repair_requires_all_five_confirmed_bars():
    from scripts.analysis.repair_stock_history_gaps import aggregate_minutes
    start = pd.Timestamp("2026-03-01T00:00Z")
    rows = [[str(int(t.timestamp()*1000)), "100", "102", "99", "101", "2", "2", "200", "1"]
            for t in pd.date_range(start, periods=5, freq="min")]
    result = aggregate_minutes(rows, start)
    assert result["volume"] == 10 and result["volume_quote"] == 1000
    assert result["open"] == 100 and result["close"] == 101
    assert aggregate_minutes(rows[:-1], start) is None
    rows[-1][-1] = "0"
    assert aggregate_minutes(rows, start) is None
