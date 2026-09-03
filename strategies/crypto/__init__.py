"""Virtual-currency strategy boundary.

The data, modeling, and backtest helpers under ``scripts`` are shared
infrastructure for this strategy.  The namespace keeps the asset-class split
explicit without duplicating the large Binance archive or trained models.
"""

SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
DATA_LAYOUT = "data/klines/*.csv.gz + data/metrics/*.csv.gz"
