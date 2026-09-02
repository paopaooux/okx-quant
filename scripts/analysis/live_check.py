"""Execution stress test: does the edge survive the two things a backtest fakes?

1. FILL TIMING.  The backtest fills at the next bar's open.  Live, the decision is
   made at the bar close and the order lands seconds later -- and if anything goes
   wrong (a missed websocket tick, a rate limit, a restart) it lands a full bar
   late.  `delay` walks that: 0 = the backtest's assumption, 1 = a whole 15m late.

2. STOP FILLS.  The backtest assumes the stop fills AT the barrier price the
   moment high/low touches it.  A stop-market does not: it crosses into a book
   that is moving away, and on a 15m crypto candle the touch is often the start of
   the move, not the end.  `sl_slip_bps` charges that, on stop fills only -- the
   take-profit rests as a limit and is not penalised.

Cost is the measured OKX round trip from costs_live.py, not the assumed 10.0.
"""
import numpy as np, pandas as pd
from scripts.backtest import backtest, portfolio as pf, asym

YRS = 3.39
COST = 10.2                       # OKX Lv1 all-taker + half-spread + funding


def run(oos, px, tail, tp, sl, delay, slip):
    tr, _ = asym.trades(oos, px, "c", tail, "both", tp, sl,
                        cost=COST, delay=delay, sl_slip_bps=slip)
    s, p = backtest.summarise(tr, YRS), pf.stats(tr)
    return s, p


def main():
    oos, px = asym.load("c")
    for tail, tp, sl in ((0.005, 0.4, 1.0), (0.02, 0.4, 1.0), (0.02, 1.0, 1.0)):
        print(f"\n=== c tail={tail}  TP={tp} SL={sl}  cost={COST}bps ===")
        print(f"{'延迟':>5}{'止损滑点':>9}{'笔数':>6}{'胜率':>7}{'净bps':>8}{'t':>7}"
              f"{'持仓h':>7}{'年化':>8}{'回撤':>8}{'夏普':>7}")
        for delay in (0, 1, 2):
            for slip in (0.0, 5.0, 10.0, 20.0):
                s, p = run(oos, px, tail, tp, sl, delay, slip)
                print(f"{delay:>4}b{slip:>8.0f}{s['trades']:>6}{s['winrate']*100:>6.1f}%"
                      f"{s['net_bps']:>+8.1f}{s['t_stat']:>+7.2f}{s['hold_h']:>7.1f}"
                      f"{p['cagr']*100:>+7.1f}%{p['max_dd']*100:>+7.1f}%{p['sharpe']:>7.2f}")


if __name__ == "__main__":
    main()
