# Daily Sharpe And September Shared Replay

Research extension, 2026-09-28. No live configuration, orders, deployments,
leverage, model retraining or universe changes. These are retrospective
simulations and actual closed-trade attribution, not interchangeable measures.

## Method And Coverage

All weights are per-entry fractions of then-realized equity, with two shared
stock/crypto slots. Weights are 20/25/30/35/40%; stock entries remain capped at
two per UTC day. Trade identity is fixed across weights and cost pairs within
each scope/delay. Costs are round-trip assumptions: ordinary 44bp stock/10bp
crypto, higher 68bp/16bp. They are not fee-tier estimates or guarantees of fills.

Sharpe uses full UTC midnight-to-midnight marked equity returns, arithmetic
mean divided by sample standard deviation (ddof=1), times sqrt(365), risk-free
rate zero. Empty/idle days remain; partial first/last days do not. Missing
midnight marks fail instead of being filled. A final midnight closes the prior
day, not an extra zero-return day. Zero variance or fewer than two daily returns
has undefined Sharpe. No serial-correlation correction is claimed.

The existing 90 sizing cases now include Sharpe and saved daily-return series.
Their original returns, drawdowns, trade counts and all other old summary
fields reproduce unchanged. Historical Sharpe has 183 full days; July-August
has 62; September 1-25 stock-only has 25.

The new September shared replay runs **September 4 00:00 UTC through September
26 00:00 UTC**, i.e. September 4-25, 22 full days. September 4 is the first full
UTC date after archived predictions begin. Stock history ends September 25.
This is not September 1-30, nor a through-September-28 account replay.

Stocks use the corrected reference-market calendars and the existing funded
baseline/5m/10m stock candidate files. Crypto uses first-observed archived
production sides, original widths and signal deadlines, not newly generated
predictions or the old Binance-label history. Entry waits for the next 15m
open at/after observation; complete price coverage through the known deadline
is required. Intrabar stop/target conflicts use stop-first, with adverse stop
gaps. The frozen crypto market files have no missing bars in this common window.

There are 100 recorded non-flat observations overall, of which 11 are excluded
by the common known-horizon boundary and 89 compete with stock candidates.
Archive logging gaps remain, including September 18-19; missing predictions
are not invented. Later same-bar revisions are not replayed. The standalone
crypto control reproduces all six old replay entry/exit identities and gross
returns. Shared capacity then accepts five crypto positions, not automatically
those six. All observations are offered to the allocator, not a preselected
crypto-only sleeve. Shared occupancy blocks the early September 6 LINK opportunity.

Start with cash; entries whose full known horizon reaches the end are excluded
regardless of whether an early stop/target could have closed them. Pre-window
positions and terminal open positions are not reconstructed. Actual historical
deployment rules/bugs and order failures are not replicated.

## Historical Sharpe

March 3 09:30 to September 3 14:30 UTC, shared two slots, higher costs, no extra
stock delay; funding omitted. Sharpe excludes the partial endpoint days.

| Per Position | Return | Marked DD | Daily Sharpe |
| --- | ---: | ---: | ---: |
| 20% | +37.58% | -8.61% | 2.642 |
| 25% | +48.28% | -10.79% | 2.643 |
| 30% | +59.51% | -12.97% | 2.644 |
| 35% | +71.26% | -15.15% | 2.644 |
| 40% | +83.54% | -17.33% | 2.645 |

Sharpe barely changes with weight. With 5m stock delay it is 1.146-1.156; with
10m it is 1.190-1.201. Higher exposure raises dollars at risk, not the signal's
predictive edge. These crypto labels are not the September live-feature model.

## September Combined Replay

September 4-25, higher costs, no extra stock delay. Stock funding is included;
crypto funding is **omitted**, so these are not fully funded net returns.

| Per Position | Return | Marked DD | Daily Sharpe |
| --- | ---: | ---: | ---: |
| 20% | +0.227% | -8.33% | 0.274 |
| 25% | +0.243% | -10.31% | 0.277 |
| 30% | +0.243% | -12.26% | 0.280 |
| 35% | +0.226% | -14.17% | 0.283 |
| 40% | +0.193% | -16.04% | 0.286 |

All five cases have 38 trades: 33 stock and five crypto, 17 winners (44.74%).
At 20%, stock contribution is -1.231 percentage points and crypto +1.458.
At 40%, these are -2.653 and +2.846. Shared occupancy changes which stocks enter:
their contributions cannot be obtained by adding independent sleeve returns.
There is no meaningful net-return gain here from doubling the per-entry weight,
despite nearly doubling the sampled marked drawdown. The small difference
between 25% and 30% is not evidence of an optimized position size.

Ordinary costs, same dates and no extra delay:

| Per Position | Return | Marked DD | Daily Sharpe |
| --- | ---: | ---: | ---: |
| 20% | +1.89% | -7.46% | 1.486 |
| 25% | +2.33% | -9.24% | 1.489 |
| 30% | +2.75% | -11.00% | 1.492 |
| 35% | +3.15% | -12.73% | 1.495 |
| 40% | +3.55% | -14.43% | 1.498 |

Higher-cost stock-delay sensitivity, with the same crypto observation timing:

| Per Position | 5m Return | 5m DD | 5m Sharpe | 10m Return | 10m DD | 10m Sharpe |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 20% | +3.01% | -8.05% | 2.174 | +1.16% | -8.36% | 0.917 |
| 25% | +3.72% | -9.98% | 2.175 | +1.40% | -10.36% | 0.919 |
| 30% | +4.42% | -11.87% | 2.176 | +1.63% | -12.32% | 0.921 |
| 35% | +5.10% | -13.73% | 2.176 | +1.84% | -14.24% | 0.923 |
| 40% | +5.77% | -15.56% | 2.177 | +2.02% | -16.13% | 0.924 |

These scenarios have 38 trades each, with win rates 50.00% (5m) and 47.37%
(10m). Delay can improve particular entries or alter admission paths in this
short sample. It is not evidence for intentionally waiting 5m: the same delay
reduced the longer historical results substantially. No best delay was selected.

All September Sharpes have only 22 daily returns. They are unstable sample
annualizations, not measured full-year performance or independent validation.
Positive arithmetic-mean Sharpe can coexist with a negative compounded return
because of volatility drag; it does not mean a losing book earned cash profit.

## Stock-Only Comparators

Both keep two slots and include stock funding, higher costs, no extra delay:

| Per Position | Sep 1-25 Return | Sep 1-25 Sharpe | Matched Sep 4-25 Return | Matched Sep 4-25 Sharpe |
| --- | ---: | ---: | ---: | ---: |
| 20% | +0.399% | 0.370 | -0.097% | 0.036 |
| 25% | +0.456% | 0.373 | -0.161% | 0.040 |
| 30% | +0.495% | 0.376 | -0.241% | 0.043 |
| 35% | +0.517% | 0.379 | -0.336% | 0.047 |
| 40% | +0.521% | 0.382 | -0.447% | 0.050 |

The different start dates and cash-start inventory explain why the old
September stock-only number is not the new matched control. The latter has
33 trades, with 13 winners (39.39%). It is separate from actual executed stocks.

## Actual Account Attribution

Frozen September 28 audit: completed positions opened September 4 onward and
closed through September 25. Actual exchange fees/funding are included, current
open positions excluded. Units are USDT, not percentage returns.

| Asset | Closed Positions | Wins | Net USDT |
| --- | ---: | ---: | ---: |
| Stocks | 42 | 16 | -4.014648 |
| Crypto | 6 | 5 | +2.252419 |
| Total | 48 | 21 | -1.762229 |

The simulated two-slot hybrid differs in rules, entries, sizes, available
capacity and execution from the actual multi-regime account. The simulation's
small gain does not rewrite the actual loss or prove what the account would
have earned. No actual account Sharpe is claimed: completed PnL alone is not a
cash-flow-adjusted daily equity series, and old demo-scale snapshots cannot be
concatenated into current live NAV.

## Validation And Artifacts

- `results/shared_two_slot_weights_20260928/`: all 90 original cases plus Sharpe.
- `results/september_shared_weights_20260928/`: 60 new shared/paired cases with
  ledgers, five-minute equity/exposure, daily returns, contributions, admission
  rejections, price/signal coverage, actual closed summary and source hashes.
- The crypto-only six-trade identity/gross-return control passes; 20% resizing
  reproduces admitted book notionals and net returns. All weights retain the
  same trade/win counts per scenario and cost pair.
- Independent output checks reconcile all 150 summary/ledger/equity/daily-return
  cases and pass all 1,006 input/code hash checks across both manifests.
- 228 tests pass; two pre-existing calendar dependency warnings remain.

Reproduce with frozen local inputs:

```bash
.venv/bin/python -m scripts.analysis.shared_two_slot_weights
.venv/bin/python -m scripts.analysis.september_shared_weights
.venv/bin/python -m pytest -q
git diff --check
```

Remaining limits: already-inspected samples; missing live features and archive
records; approximate stock funding marks and omitted crypto funding; OHLC
intrabar ambiguity; fees/funding deferred for sizing; fractional lots; no
exchange margin rejection, liquidation or size-dependent impact. Historical
drawdowns are not loss limits. No automatic live weight change is supported by
this extension.
