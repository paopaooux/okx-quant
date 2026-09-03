"""Compatibility exports for the virtual-currency backtest."""

from scripts.backtest.backtest_dir import simulate
from scripts.backtest.portfolio import curve, stats

__all__ = ["simulate", "curve", "stats"]
