"""Canonical stock-strategy backtest entry point."""

import os

from scripts.run_stock_strategy import _args
from .run_strategy import main

if __name__ == "__main__":
    args = _args()
    if args.stock_alpha_root:
        os.environ["STOCK_ALPHA_ROOT"] = args.stock_alpha_root
    if args.data_dir:
        os.environ["STOCK_ALPHA_DATA_DIR"] = args.data_dir
    if args.result_dir:
        os.environ["STOCK_ALPHA_RESULT_DIR"] = args.result_dir
    raise SystemExit(main())
