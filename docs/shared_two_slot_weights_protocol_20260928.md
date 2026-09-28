# Shared Two-Slot Position Weight Comparison

Requested weights: 20% control, 25%, 30%, 35%, 40% of realized equity per
entry. Exactly two shared slots, no leverage change, no strategy/threshold or
universe selection. Research only: no live configuration or order changes.

Use the frozen admitted books from the capacity-confirmation study. Weight
does not enter those admission rules. Recompute each notional from then-current
realized equity, settling completed trades before the next entry. Never scale
the old cumulative return or its old notionals by a constant multiplier.
Preserve the parent's deferred fee/funding sizing convention and causal marks.

Compare all five weights at both cost pairs (stocks/crypto 44/10 and 68/16bp),
with the existing 0/5/10-minute stock entry delays. Windows:

- Shared historical: March 3-September 3, without funding.
- Shared July-August: actual stock funding approximation, no crypto funding.
- Stock-only September 1-25: two stock positions, stock funding included;
  this is a distinct comparator when live crypto is unavailable, not a combined
  September portfolio with fabricated crypto signals.

Report return, 5m marked drawdown, recovery, worst day/month, average holding,
win rate, and entry-notional/current-mark-notional exposure versus marked
equity. A 1x headroom proxy is diagnostic only: no exchange mark history,
account-mode calculation, tiers, order loss, liquidation, lot rounding or
size-dependent market impact is modeled. A negative proxy must be visible;
it must not be silently treated as available margin or a feasible fill.

Verify the 20% controls against saved results and preserve identical admitted
trade identities across weights/costs for each scenario/delay. Write all
results and input/code hashes. Do not pick an automatic live percentage from
the highest retrospective return. Existing crypto feature and sample-selection
limitations remain, and increased weight cannot repair a negative trade edge.
