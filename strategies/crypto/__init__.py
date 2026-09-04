"""Virtual-currency strategy boundary.

The data, modeling, and backtest helpers under ``scripts`` are shared
infrastructure for this strategy.  The namespace keeps the asset-class split
explicit without duplicating the large Binance archive or trained models.
"""

SYMBOLS = (
    "ADAUSDT", "BNBUSDT", "BTCUSDT", "DOGEUSDT",
    "ETHUSDT", "LINKUSDT", "SOLUSDT", "XRPUSDT",
)
DATA_LAYOUT = "data/klines/*.csv.gz + data/metrics/*.csv.gz"
