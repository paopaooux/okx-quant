# Stock And Portfolio Robustness Research

Research date: 2026-09-28. No deployment, live-universe change or account action.
All returns below are simulated account returns, not actual account PnL.

## Decision

There is research space, but no execution-robust production promotion from this
round. The most useful findings are:

1. Correct the return expectation before optimizing it. Small changes between
   the signal close and the next open can change whether a 3% stop is hit.
2. Displacement <=8%, a two-stock cap, and a two-slot shared book are distinct
   candidates. Their favorable historical rows do not survive every delay test.
3. Fixed take profit can raise win rate while reducing net profit. More
   restrictive liquidity rules and earlier exits did not improve this sample.
4. Cost/execution measurement deserves priority over more parameter searches.
   Keep candidate rules in research/shadow comparison, not automatic promotion.

## Design And Coverage

- 160 reviewed contracts, 5,069,687 five-minute candles, 1 internal missing bar.
  Snapshots were merged only after overlapping OHLCV values matched.
- 1,207 independently priced corrected-calendar events; no candidate was lost
  to a missing price path in this run. The old 1,049 historical candidates were
  reproduced by the original pricing convention, including exits and returns.
- Historical window: March 3 09:30 UTC to September 3 14:30 UTC. Recent window:
  September 11 through September 25. Development, June-July, August and all
  September 1-25 are also reported. These windows overlap and are not independent.
- Original universe retained. No stock removed because of its historical PnL.
- Default: 6% trigger, 3% stop, reference-market open+60m exit, 30h hold cap,
  three stock positions, two stock entries per UTC day, 20% per-position weight.
- Next-bar-open entries; gap-aware stops; stop-first ambiguous barriers; timed
  exits at deadline-bar open. Only post-entry candles enter the exit scan.
- Stock round-trip costs: 44bp and 68bp. These are assumptions, not measured
  account fee rates. Frozen crypto costs: 10bp and 16bp respectively.
- Common preknown deadlines purge fold boundaries, even for trades that would
  have stopped early. Every period begins flat. Five-minute-close marked risk
  is reported, not merely realized-equity drawdown.
- All dates had already been inspected. Chronological selection is NOT an
  untouched out-of-sample test. Current-listing universe/survivorship limits remain.

The original preregistered protocol remains byte-identical to its recorded hash.
The two follow-up protocols explicitly disclose that initial results were seen.

## Execution Bridge

Stock-only historical returns, no funding:

| Convention | Trades | 44bp Return | 68bp Return |
| --- | ---: | ---: | ---: |
| Previously reported corrected calendars | 241 | +33.36% | +18.84% |
| Same old fills, common preknown boundary purge | 239 | +34.05% | +19.58% |
| Next-open entry, exact stop, old timed exit | 238 | +23.17% | +9.90% |
| Also use deadline-bar open | 238 | +23.02% | +9.77% |
| Also gap-aware stops | 238 | +23.02% | +9.77% |

The same-boundary execution difference is therefore -9.80 percentage points
at 68bp, not simply the difference from the old 241-trade table. Admission and
compounding change when stop times change. On the old fixed book and notionals,
entry repricing contributes -7.61 points, timed exits -0.14, and stop-gap fills
zero in this particular sample. Gap-aware treatment remains necessary despite
its zero realized contribution here.

Examples include KORU June 28 and July 8, MU March 19 and BMNR June 15: small
entry differences move the stop barrier enough to change a later profitable
deadline exit into an earlier stop. These are not claims of enormous observed
live slippage. Do not delete these names after seeing their contribution.

The recent old-fill control was also reproduced: +4.00% at 44bp, +2.91% at
68bp. The new-fill recent baseline is +4.15% / +3.06%, illustrating that the
effect is sample-dependent, not an automatic haircut of a constant size.

## All Predeclared Variants

Higher-cost scenario, 68bp. Historical returns exclude funding. The final
column is a different, overlapping July 1-September 25 window with available
stock funding included using the notional approximation described below.

| Variant | Historical Trades | Win % | Historical Return % | Marked DD % | Jul-Sep Funded Return % |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 238 | 43.28 | +9.77 | -12.33 | -0.56 |
| Basic liquidity | 223 | 43.05 | +8.63 | -13.52 | -5.44 |
| Strict liquidity | 207 | 39.13 | -5.89 | -20.21 | -14.43 |
| Listing age >=7d | 218 | 42.20 | -0.48 | -18.51 | -5.74 |
| Reference open within 12h | 167 | 39.52 | -3.09 | -22.32 | -10.18 |
| Displacement <=8% | 234 | 44.87 | +19.26 | -7.66 | +6.59 |
| Fresh entry price guard | 232 | 43.53 | +9.40 | -13.82 | -4.91 |
| Exit at reference open | 238 | 41.60 | -9.49 | -19.25 | -7.97 |
| Exit open+15m | 238 | 44.54 | -0.30 | -17.93 | -7.78 |
| Hold cap 12h | 239 | 45.61 | -1.11 | -17.02 | -7.07 |
| Half-anchor take profit | 239 | 58.16 | -2.58 | -10.95 | -9.50 |
| Fixed 3% take profit | 239 | 59.83 | -3.76 | -12.00 | -9.59 |
| Stop 2% (diagnostic) | 240 | 36.25 | +2.43 | -10.74 | -0.41 |
| Overnight only | 164 | 37.80 | -4.40 | -18.97 | -4.36 |
| One stock slot | 182 | 37.36 | -8.12 | -16.76 | +1.00 |
| Two stock slots | 227 | 43.61 | +15.72 | -8.55 | +3.20 |
| Hynix underlying dedup | 238 | 43.28 | +9.91 | -12.33 | +3.51 |
| Max two same-direction stocks | 234 | 43.16 | +11.77 | -10.66 | +0.08 |
| Long only (diagnostic) | 140 | 37.86 | -8.72 | -23.65 | -19.39 |
| Short only (diagnostic) | 186 | 44.62 | +19.07 | -8.02 | +10.23 |
| Liquidity + open+15m exit | 223 | 43.05 | -2.66 | -21.22 | -13.77 |
| Liquidity + half-anchor target | 223 | 57.85 | -4.27 | -13.37 | -16.09 |
| Combined entry-quality rules | 163 | 41.72 | +2.56 | -18.81 | -9.34 |
| Delay 5m | 238 | 41.18 | -3.97 | -21.76 | -6.55 |
| Delay 10m | 238 | 41.60 | +3.85 | -16.82 | -3.39 |
| Delay 5m + fresh guard | 193 | 42.49 | +3.22 | -20.68 | -11.05 |

Liquidity remains an execution feasibility concern. These failed volume
thresholds do not justify buying through inadequate depth; they show that the
particular thresholds tested did not supply a robust return improvement.
Short-only results are a direction diagnostic, not authorization to fit a new
permanent direction restriction after seeing these returns.

## Funding And Costs

The public funding-history download contains 39,938 settlements across 160
contracts. All 849 July-onward candidate paths have coverage in every tested
variant. Missing early-month histories were not filled with zero or inferred
from later observations. March-June results must not be called funding-net.

Use actual `realizedRate`, but the last completed perpetual close approximates
the historical settlement mark notional. Funding is applied at settlement for
marked equity and at position exit for the original realized-equity sizing
convention. This is not a reconstruction of account bills or exact mark prices.

At uncertain stop/target/deadline settlement boundaries, include debits and
exclude credits. Baseline July-September has three such settlements; the
fixed-notional optimistic-versus-conservative difference is only 0.0021 points.

Baseline, July-September, same 151 trades:

| Cost | Without Funding | With Approximate Funding |
| --- | ---: | ---: |
| 44bp | +6.75% | +6.90% |
| 68bp | -0.71% | -0.56% |

At 68bp, funding contributes +0.141 points before compounding feedback. Trading
cost and execution assumptions dominate that funding effect in this sample.

## Selection And Uncertainty

The fixed expanding-window selector trains before each month, uses 68bp costs,
purges common horizons and includes cash. Its frozen following-month outcomes:

| Test Month | Selected Rule | Selected Return | Baseline Return |
| --- | --- | ---: | ---: |
| June | Entry quality | +3.12% | +2.14% |
| July | Fresh price guard | -9.40% | -8.61% |
| August | Cash | 0.00% | +10.03% |
| September 1-25 | Displacement <=8% | -1.26% | -1.26% |

Compounded selected return: -7.75%; baseline: +1.42%. Do not replace a stable
rule with this adaptive selector. This test enforces temporal discipline but
does not create a fresh holdout from previously viewed prices.

Ten thousand paired seven-calendar-day block resamples were run, seed 20260928.
Historical return advantages versus baseline, in percentage points:

| Rule | Observed Advantage | Ordinary 95% Interval | Approx. Family-Adjusted Interval |
| --- | ---: | --- | --- |
| Displacement <=8% | +9.49 | [+1.26, +21.36] | [-2.23, +30.15] |
| Two stock slots | +5.95 | [-3.18, +18.81] | [-9.52, +27.85] |

The family adjustment uses Bonferroni tails across 25 non-baseline rules.
Extreme tails have only about ten simulated observations; these are approximate
exploratory intervals, not guaranteed coverage under changing market regimes.
Both intervals include zero after adjustment. The displacement rule's own
historical absolute-return interval is [-5.92%, +51.63%].

No single instrument contributes more than 50% of positive PnL in the main
candidates. Historical top-positive shares: baseline 12.73%, displacement cap
11.63%, stock cap two 10.08%. Fixed-book return after subtracting the largest
symbol contribution remains +3.10%, +12.54%, +9.85%, respectively. This is an
influence diagnostic, not a tradable rule to remove that symbol.

## Candidate Delay Confirmation

No new threshold grid was searched. Exact existing rules were tested at the
same delays. July-September, 68bp with approximate stock funding:

| Rule | No Delay | Delay 5m | Delay 10m |
| --- | ---: | ---: | ---: |
| Baseline | -0.56% | -6.55% | -3.39% |
| Displacement <=8% | +6.59% | -2.34% | +1.15% |
| Two stock slots | +3.20% | -5.01% | -5.73% |

Displacement <=8% still has an encouraging comparative signal, but fails an
absolute-profit claim under the 5m delayed covered window. The two-stock cap
is even more delay-sensitive. Delay durations are scenarios, not estimates of
the live bot's measured distribution. Recent September 11-25 results have only
22 admitted trades per main candidate, too little to override these failures.

## Crypto And Shared Capacity

Crypto predictions and barrier outcomes remain frozen, from the original
Binance-based archive. Its recorded signal-bar timestamp is shifted +15m to
the actual entry open; entry prices are checked against that open. The frozen
label scanner's extra last-bar convention is preserved and a common signal+50
bars horizon is purged. No retraining or live missing-feature repair is implied.

Historical, 68bp stock / 16bp crypto, no funding, fixed 20% per trade:

| Rule | Stock-Only Return | Shared Five-Slot Return | Shared Marked DD |
| --- | ---: | ---: | ---: |
| Baseline | +9.77% | +16.32% | -12.87% |
| Displacement <=8% | +19.26% | +26.38% | -8.38% |
| Stock cap two | +15.72% | +22.64% | -9.45% |

Frozen crypto alone has 58 trades, +5.96%, marked DD -7.31%. This is not a
forecast for a live crypto model with missing features. Matched two-stage and
unified admission are identical in the five-slot controls; this ceases to hold
when shared capacity is reduced.

All 1-5 capacities were tested for stocks, crypto and shared allocation, with
both fixed 20% and equal 1/slots weights, both costs. Fixed-20% returns at the
higher costs:

| Scope | 1 Slot | 2 Slots | 3 Slots | 4 Slots | 5 Slots |
| --- | ---: | ---: | ---: | ---: | ---: |
| Stock only, stock cap=slots | -8.12% | +15.72% | +9.77% | +6.98% | +6.98% |
| Frozen crypto only | +5.06% | +6.75% | +7.09% | +5.96% | +5.96% |
| Shared, stock cap stays 3 | +7.82% | +37.58% | +24.20% | +18.24% | +16.32% |

Shared two slots is a separate research candidate, not the stock-cap-two row.
At fixed 20% it has 264 trades and marked DD -8.61%. Its retrospective paired
family-adjusted advantage interval versus five shared slots is approximately
[+2.38, +57.86] points across the separate 24-comparison capacity family. That
does not resolve uncertainty in the frozen crypto model, costs or execution.

Do not silently resize each shared-two-slot position from 20% to 50%: the latter
produces +109.61% but DD -21.67% in this inspected sample. A one-slot stock book
at 100% per trade loses 43.82% with DD -64.99%. Fewer positions is not the same
as lower risk when position weights increase. Monthly returns, recovery times
and unrecovered drawdown flags are saved for every capacity configuration.

### Shared-Capacity Confirmation

All five capacities were subsequently checked with both unified and two-stage
allocation, 0/5/10m stock delays and the fixed 20% position weight. Crypto fills
remain frozen; crypto funding is omitted. Selected comparisons below report
July-August only, because the frozen crypto archive does not span September 25.

| Shared Capacity, Unified | No Delay | Stock Delay 5m | Stock Delay 10m |
| --- | ---: | ---: | ---: |
| Two slots, with stock funding | +9.73% | -0.58% | -1.95% |
| Five slots, with stock funding | +5.98% | -2.59% | +1.61% |

Two slots' marked DD rises from -8.57% to -18.39% / -21.71% in the delayed
covered window. Under the old two-stage preselection, historical no-delay
two-slot return is +28.02%, versus unified +37.58%; rejected shadow trades can
consume sleeve opportunities in the former. This is why allocator conventions
must be matched, not mixed into an apparent signal improvement.

The shared-two-slot result merits further shadow comparison, but its execution
and source-model assumptions fail a claim of robust live improvement. If the
live crypto sleeve has no valid signals, this benefit cannot be extrapolated
from the historical mixed book; the system behaves more like stock-only.

## Supported Actions And Limits

- Use the next-open/common-horizon study as the research comparator. Do not
  reuse earlier ideal-fill/US-clock headline returns as execution forecasts.
- Measure live decision-to-order, order-to-fill, spread, depth and realized
  slippage against the correct reference anchor before choosing a new rule.
- Keep displacement <=8% and shared capacity two as explicit shadow candidates;
  retain the baseline control and unchanged 20% sizing. They are different
  experiments, not a combined optimized live strategy.
- Retain correctness protections such as market calendars and verified
  same-underlying limits for their risk purpose, not a promised profit increase.
- Do not promote fixed take profit, stricter volume thresholds, blanket earlier
  exits, recent-only weekend rules or the monthly best-rule selector from this run.
- Resolve the crypto training/live feature mismatch before interpreting frozen
  crypto allocation gains as an available live opportunity.

None of these simulations reconstruct historical order books, lot rounding,
available margin, liquidations, mark-price funding cash flows or intrabar fill
times. Five-minute marked drawdowns miss intrabar extremes. Funding is only
partially available, and the current universe is not a historical security
master. Exact live profitability remains unproven. No strategy or account
configuration was changed by this research pass.

## Artifacts And Reproduction

Every directory below includes machine-readable results; trade lists are
retained, including failed hypotheses. Paths are relative to the project root.

| Directory | Evidence |
| --- | --- |
| `results/stock_robustness_20260928/` | 312 variant/period/cost rows; 104 train rows; 8 frozen fold outcomes; hashes/coverage |
| `results/stock_execution_audit_20260928/` | 18 bridge rows; 1,049 old candidates reproduced; per-event and fixed-book deltas |
| `results/stock_funding_robustness_20260928/` | 520 matched funded/unfunded rows; raw-rate coverage and contribution |
| `results/stock_portfolio_robustness_20260928/` | 114 portfolio comparisons; 60 capacity/sizing/cost rows |
| `results/stock_robustness_uncertainty_20260928/` | 286 main bootstrap/concentration rows each; 30 capacity rows; 210 capacity-month rows |
| `results/stock_robustness_confirmation_20260928/` | 126 fixed-candidate delay/cost/period rows |
| `results/stock_capacity_confirmation_20260928/` | 60 shared-capacity/delay/allocator checks, monthly and sleeve contribution tables |
| `data/research_stock_funding_20260928/` | Frozen raw public realized funding rates and download manifest |

With the frozen local inputs present:

```bash
.venv/bin/python -m scripts.analysis.stock_robustness_study
.venv/bin/python -m scripts.analysis.stock_execution_audit
.venv/bin/python -m scripts.analysis.stock_funding_robustness
.venv/bin/python -m scripts.analysis.stock_portfolio_robustness
.venv/bin/python -m scripts.analysis.stock_robustness_uncertainty
.venv/bin/python -m scripts.analysis.stock_robustness_confirmation
.venv/bin/python -m scripts.analysis.stock_capacity_confirmation
.venv/bin/python -m scripts.analysis.verify_stock_robustness
.venv/bin/python -m pytest -q
git diff --check
```

Verification checks input hashes, original protocol hashes, summary/trade PnL
and win-rate identities, costs/funding, causal timestamp ordering, known horizon
coverage, capacity limits, fold compounding and required table sizes. These are
integrity checks, not proof of future alpha. Causal-price/filter/mark tests and
the full project suite are also required; the resulting verification manifest
is `results/stock_robustness_20260928/verification.json`.

Final executed checks: **1,208 trade/summary pairs and 1,337 source/code/result
hashes verified; 201 tests passed; `git diff --check` passed.** The two test
warnings are existing third-party calendar deprecations, not failed assertions.

Completion audit: frozen coverage, legacy controls, all 26 declared rules,
both costs, all named periods, expanding frozen folds, funding coverage/paired
controls, execution attribution, concentration/bootstrap uncertainty, three
portfolio scopes, both sizing conventions and 1-5 capacities, recovery risk,
follow-up delay/allocator checks, source hashes, trade lists and the report
are present and verified. Production promotion gates remain unsatisfied; that
is a research conclusion, not an omitted test or an authorization to deploy.
