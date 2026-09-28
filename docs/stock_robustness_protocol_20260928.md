# Stock Robustness Research Protocol

Written before generating this round's variant results. Research only; do not
deploy, alter the live universe, submit orders, or retune the crypto model.

## Questions

1. Does the corrected-calendar edge survive next-bar-open entries, gap-aware
   stops, delayed fills, higher costs, and historical funding where available?
2. Can simple pre-entry quality rules or earlier exits improve multiple time
   periods without selecting individual stocks by their past returns?
3. What happens to marked drawdown, concentration, trading frequency and the
   combined portfolio, rather than just win rate or total return?

## Fixed Research Design

Use the existing 160 reviewed calendar mappings. Merge the isolated recovered
February-September OHLC archive with the frozen September 11 snapshot and the
September 25 snapshot. Reject conflicting overlapping prices. Do not download
new live prices into the research samples or manufacture missing candles.
Keep the 6% trigger, 3% default stop, 30-hour default cap, two stock entries per
UTC day, three stock slots, and 20% per-position allocation unless the named
variant explicitly changes a capacity or exit rule.

First reproduce the corrected historical/recent controls. For new execution
tests enter at the next bar's open after observing the signal-bar close. Scan
only post-entry candles. Gap-through stops fill at the adverse open, with stop
priority when an OHLC bar touches both stop and target. A timed exit executes
at the deadline bar's open, without using its later extrema. These are still
assumed fills, not reconstructed historical books. Test round-trip costs of
44 and 68bp, and zero/5/10-minute execution-delay scenarios explicitly.

Predeclared one-change variants: trailing 24h volume >=150k and signal-bar volume
>=1.4k; a stricter 500k/5k liquidity rule; listing age >=7 days; time to reference
open <=12h; observed displacement <=8%; fresh entry price retaining >=6%
displacement and <=30bp adverse move; exit at reference open; exit 15 minutes
after open; cap holding at 12h; take profit at half the original displacement
back toward the anchor; fixed 3% take profit; 2% stop (risk sensitivity); skip
weekend/holiday windows; stock capacity 1 or 2 while retaining 20% weights;
existing verified Hynix-underlying deduplication; at most two same-direction
stock positions. Long-only and short-only are diagnostics, not automatic
selection candidates. Predeclared combinations: basic liquidity with open+15m,
basic liquidity with half-displacement take profit, and the <=12h/<=8%/fresh-price
entry-quality rules. No stock deletion based on its realized PnL.

Report the original March 3-September 3 and September 11-25 windows. Also use
expanding monthly development windows beginning March 3: before June 1, July 1,
August 1 and September 1, respectively; each selected rule is frozen for the
following June, July, August and September 1-25 replay. All these historical
prices have already been inspected in this thread; this is chronological
selection discipline, NOT a claim of untouched out-of-sample evidence.
Use a common, knowable baseline deadline to purge positions crossing fold
boundaries, regardless of whether they later stop early.

For a development selection require >=20 trades and >=10 entry days, positive
net expectancy at 68bp, and marked drawdown no worse than 1.25 times baseline
(minimum allowance 2 percentage points). Rank by return plus signed marked
drawdown at 68bp, with ties resolved by the fixed variant order. Include cash
and the baseline as explicit alternatives. Never revise the rule after seeing
the next month; report all failed variants too. Inspect day-cluster bootstrap
uncertainty and symbol concentration, not just the selected maximum.

A candidate is research-worthy only if it improves more than the development
window, survives the higher cost, has >=30 validation trades across >=10 entry
days, and is not explained by one instrument supplying >50% of positive
contribution. A recent window with <20 trades is inconclusive; a negative
recent/stress result does not justify immediate production promotion.

## Funding and Portfolio Checks

Fetch public realized funding-rate histories into a separate frozen directory.
Use only actual available coverage; do not infer early-month rates from later
ones. Rates are genuine settlement rates, but historical mark notionals are
approximated with completed perpetual candles. Flag settlement-time ambiguity
inside stop bars and report a conservative debit/no-credit boundary treatment.
Funding outcomes must not become entry features. Missing coverage is excluded
with a paired unfunded control, never silently set to zero.

For promising rules and operational controls, replay the historical crypto
signals as a frozen comparator. Separate the original two-stage allocator from
any unified-allocation diagnostic, and account for crypto's known next-bar
entry timestamp. Capacity changes are risk-budget experiments, not new alpha.
No result addresses the unresolved difference between crypto training inputs
and live missing-feature inputs.

## Deliverables

Frozen input hashes, overlap/gap audits, variant definitions, all results and
trade lists, monthly/fold selections, tests for causal prices/filters, and a
report distinguishing supported improvements, rejected hypotheses, and limits.
No promotion merely because one retrospective table looks better.
