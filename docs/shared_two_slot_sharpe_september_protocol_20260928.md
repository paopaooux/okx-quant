# Sharpe And September Shared-Book Addendum

Requested after the original weight comparison. Its outcomes were already
inspected; this is a retrospective extension, not a new holdout. The original
weight protocol remains unchanged. No production configuration, deployment,
account, orders, thresholds or universe selection is authorized by this study.

## Daily Risk

Add annualized daily Sharpe to all 90 original sizing rows without changing
their admissions, compounding, returns or drawdowns. Use positive marked equity
at consecutive UTC midnights: arithmetic daily returns, sample standard
deviation (ddof=1), risk-free rate zero, annualization sqrt(365). Calendar and
idle days stay in the sample because the instruments are 24/7 perpetuals.

Exclude the first/last partial calendar day. If the portfolio begins at
midnight, its initial boundary is pre-trade equity 1, including entry costs in
the first day's return. Do not manufacture a zero-return day at the final
midnight. Missing boundary marks fail; fewer than two daily returns or zero
variance produce an undefined Sharpe. Save the exact daily-return series and
sample dates/counts. This is an unadjusted sample Sharpe, not an independent-
returns assumption test, forecast or confidence interval.

## September Replay

Use September 4 00:00 UTC through September 26 00:00 UTC (22 complete days).
September 4 is the first full UTC day after the archived live predictions
begin; September 25 is the final full day of the frozen stock history. Start
with cash, exclude entries whose known full holding horizon reaches the end.
No pre-window inventory or end-window open positions are reconstructed.

Stock candidates use the existing corrected market-calendar rules and funded
baseline/5m-delay/10m-delay candidate files. No further product selection.
Crypto uses only each symbol/bar's first archived production side, stop width
and signal deadline. Entry is the next full 15m open after observation; require
a closed, fresh signal and a complete candle path through its known deadline.
Use the existing stop-first, adverse-gap scanner without retuning thresholds.
Do not preselect six crypto-only trades before shared admission: all eligible
recorded candidates compete chronologically with stocks, so a rejected signal
does not consume future capacity. Same-symbol overlap is forbidden. Do not add
the old Binance frozen-label cooldown to these recorded live-bar candidates.

Report all five requested weights, both prior cost pairs (stock/crypto 44/10
and 68/16 bp round trip), and all three stock delays. Crypto has its observation
delay, but no extra stock-delay shift. Keep two shared slots and the two-stock-
entries-per-UTC-day rule; use the parent's strength/symbol tie ordering. Also
run a matched stock-only two-slot control on these exact dates. Resize each
entry from realized equity, matching the parent convention. This is 60 rows.

Stock funding is included using the existing conservative settlement model;
crypto funding is omitted, not asserted to be zero. Costs are modeled in net
returns and equity, not a claim to reconstruct actual fill prices or fee tiers.
Intrabar exit timestamps/marks, fractional lots, deferred cost/funding sizing,
no margin rejection, and missing archived predictions remain limitations.

Save source/code hashes, admission rejections, coverage, per-trade ledgers,
5m marked equity, daily returns, stock/crypto contributions, and controls. The
crypto-only identity control must reproduce the original archived-side replay
in the covered window before applying shared occupancy and fixed cost pairs.

Keep actual completed-position PnL separate, with actual fees/funding in USDT.
Do not infer actual account Sharpe, returns or drawdown from sums of completed
trades. The database includes earlier demo-scale equity and has no reconciled
cash-flow-adjusted full-month equity supplied for this task. September sample
Sharpes use only 22 or 25 days and are descriptive, not stable annual estimates.
