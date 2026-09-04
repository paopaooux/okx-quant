"""Run the stock-perpetual off-hours strategy."""
from __future__ import annotations

import os

from .run_strategy import main


def run() -> int:
    os.environ.setdefault("STOCK_ALPHA_DATA_DIR", "data/stocks_swap")
    os.environ.setdefault("STOCK_ALPHA_RESULT_DIR", "results/stocks_swap")
    return main()


if __name__ == "__main__":
    raise SystemExit(run())
