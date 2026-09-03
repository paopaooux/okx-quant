"""Run the migrated non-LLM tokenized-stock strategy.

The migrated local archive under ``data/stocks`` and ``results/stocks`` is used
by default.  ``STOCK_ALPHA_ROOT`` or the explicit directory variables can still
override it for alternate snapshots.
"""

from __future__ import annotations

import argparse
import os

from scripts.stocks.run_strategy import main


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stock-alpha-root", default=None,
                        help="隔壁股票归档根目录（含 data/ 与 results/）")
    parser.add_argument("--data-dir", default=None,
                        help="覆盖股票行情目录")
    parser.add_argument("--result-dir", default=None,
                        help="覆盖股票结果目录（建议回测时使用临时目录）")
    return parser.parse_args()


if __name__ == "__main__":
    args = _args()
    if args.stock_alpha_root:
        os.environ["STOCK_ALPHA_ROOT"] = args.stock_alpha_root
    if args.data_dir:
        os.environ["STOCK_ALPHA_DATA_DIR"] = args.data_dir
    if args.result_dir:
        os.environ["STOCK_ALPHA_RESULT_DIR"] = args.result_dir
    raise SystemExit(main())
