import pandas as pd
import pytest
import numpy as np

from scripts.analysis.okx_retest import coverage, _native_metrics, _barrier_candidates, QUANT_SYMBOLS, STEP
from scripts.data.build import triple_barrier


def test_coverage_reports_missing_bars_without_filling():
    start = pd.Timestamp("2026-01-01", tz="UTC")
    end = start + pd.Timedelta(minutes=45)
    frame = pd.DataFrame({"ts": [int(start.timestamp() * 1000),
                                 int((start + pd.Timedelta(minutes=30)).timestamp() * 1000)]})
    result = coverage(frame, start, end)
    assert result["requested_bars"] == 3
    assert result["missing_bars"] == 1


def test_native_metrics_keeps_missing_fields_and_maps_taker(tmp_path):
    index = pd.date_range("2026-08-27", periods=3, freq="15min", tz="UTC")
    candle = pd.DataFrame({"open": [1., 1., 1.], "high": [1., 1., 1.], "low": [1., 1., 1.],
                           "close": [1., 1., 1.], "volume": [10., 10., 10.], "volume_quote": [10., 10., 10.]}, index=index)
    (tmp_path / "metrics").mkdir()
    ts = [int(t.timestamp() * 1000) for t in index]
    pd.DataFrame({"ts": ts, "oi_ccy": [1., 1., 1.]}).to_csv(
        tmp_path / "metrics/BTCUSDT_oi_15m.csv.gz", index=False, compression="gzip")
    pd.DataFrame({"ts": ts, "sell_quote_volume": [2., 2., 2.],
                  "buy_quote_volume": [6., 6., 6.]}).to_csv(
        tmp_path / "metrics/BTCUSDT_taker_15m.csv.gz", index=False, compression="gzip")
    k, m = _native_metrics(tmp_path, "BTCUSDT", candle)
    assert m["sum_open_interest"].notna().all()
    assert m["count_toptrader_long_short_ratio"].isna().all()
    assert m["sum_taker_long_short_vol_ratio"].iloc[0] == pytest.approx(3.)
    assert k["taker_buy_quote_volume"].iloc[0] == pytest.approx(7.5)
    assert k["count"].isna().all()


@pytest.mark.parametrize("outcome", ["up", "down", "both", "timeout", "last_bar"])
def test_okx_barriers_match_frozen_label_timing(outcome):
    index = pd.date_range("2026-08-27", periods=55, freq="15min", tz="UTC")
    frame = pd.DataFrame({"open": 100., "high": 100.5, "low": 99.5}, index=index)
    # The decision candle must never participate in the execution outcome.
    frame.loc[index[0], ["high", "low"]] = [200., 1.]
    if outcome == "up":
        frame.loc[index[1], "high"] = 102.
    elif outcome == "down":
        frame.loc[index[1], "low"] = 98.
    elif outcome == "both":
        frame.loc[index[1], ["high", "low"]] = [102., 98.]
    elif outcome == "last_bar":
        frame.loc[index[49], "high"] = 102.
    width = np.full(len(frame), .01)
    _, _, held, ret = triple_barrier(frame.open.to_numpy(), frame.high.to_numpy(), frame.low.to_numpy(), width, 48)
    oos = pd.DataFrame({"symbol": ["BTCUSDT"], "dt": [index[0]], "ts": [int(index[0].timestamp()*1000)],
                        "held_c": [held[0]], "width_c": [.01], "p_up": [.9], "hi_0.01": [.8], "lo_0.01": [.2]})
    result = _barrier_candidates(oos, {symbol: frame for symbol in QUANT_SYMBOLS})
    assert len(result) == 1
    assert result.iloc[0].gross == pytest.approx(ret[0])
    assert result.iloc[0].exit_ts == index[0] + (held[0] + 1) * STEP
    assert result.iloc[0].actual_entry_ts == index[1]


def test_okx_barriers_reject_gaps():
    index = pd.date_range("2026-08-27", periods=55, freq="15min", tz="UTC")
    frame = pd.DataFrame({"open": 100., "high": 100., "low": 100.}, index=index).drop(index[5])
    oos = pd.DataFrame({"symbol": ["BTCUSDT"], "dt": [index[0]], "ts": [int(index[0].timestamp()*1000)],
                        "held_c": [48.], "width_c": [.01], "p_up": [.9], "hi_0.01": [.8], "lo_0.01": [.2]})
    with pytest.raises(ValueError, match="Noncontiguous"):
        _barrier_candidates(oos, {symbol: frame for symbol in QUANT_SYMBOLS})
