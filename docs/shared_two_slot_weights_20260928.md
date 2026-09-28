# Shared Two-Slot Position Weights

Research date: 2026-09-28. No live settings, leverage, orders or deployments
changed. All figures are simulated, not actual account returns or guarantees.

## What Was Compared

The user requested 25%, 30%, 35% and 40% per position, versus the existing 20%
research control. All cases retain exactly two shared stock/crypto slots. The
target combined entry allocation is therefore 40%, 50%, 60%, 70% or 80%, not
an automatic 50% allocation to each slot. No signal or exit rules were tuned.

Each next trade is resized using the then-current realized equity. Returns are
recompounded from individual trades, not scaled from the old headline return.
The nine frozen admission books have identical trades at every tested weight;
the original model has no weight-dependent liquidity or margin rejection.

The complete study has 90 rows: five weights, two cost pairs, three stock entry
delays (0/5/10m), and three windows. Higher cost means 68bp stock / 16bp crypto;
ordinary cost means 44bp stock / 10bp crypto, all round-trip assumptions.

Windows are:

- Shared historical: March 3 09:30 UTC to September 3 14:30 UTC, no funding.
- Shared July-August: actual stock funding rates with completed-close notional
  approximation; crypto funding omitted.
- Stock-only September 1-25: stock cap two, stock funding included. Frozen crypto
  outcomes end September 3, so no fictitious September combined signal is added.

The subsequent requested Sharpe/September extension is described below and in
`docs/shared_two_slot_sharpe_september_20260928.md`. It uses archived live crypto
sides for a separate September 4-25 shared replay, not the old Binance labels.

No extra delay means the next bar open, not a guaranteed live fill at the
signal close. Only stock entries are delayed; crypto outcomes remain frozen.

## Historical Return And Risk

No extra stock delay, fixed shared two slots, March 3-September 3:

| Per Position | Two-Slot Target | Ordinary-Cost Return | Higher-Cost Return | Ordinary-Cost DD | Higher-Cost DD |
| --- | ---: | ---: | ---: | ---: | ---: |
| 20% | 40% | +53.59% | +37.58% | -7.32% | -8.61% |
| 25% | 50% | +70.13% | +48.28% | -9.08% | -10.79% |
| 30% | 60% | +88.09% | +59.51% | -10.81% | -12.97% |
| 35% | 70% | +107.54% | +71.26% | -12.50% | -15.15% |
| 40% | 80% | +128.56% | +83.54% | -14.17% | -17.33% |

All five higher-cost cases have the same 264 trades and 46.59% win rate.
Increasing the weight supplies no new predictive edge. Higher historical
returns reflect taking more exposure to the same favorable inspected sequence.

Added daily annualized Sharpe (zero risk-free rate, complete UTC calendar days):

| Per Position | Ordinary-Cost, No Delay | Higher-Cost, No Delay | Higher-Cost, 5m | Higher-Cost, 10m |
| --- | ---: | ---: | ---: | ---: |
| 20% | 3.506 | 2.642 | 1.146 | 1.190 |
| 25% | 3.507 | 2.643 | 1.148 | 1.193 |
| 30% | 3.507 | 2.644 | 1.151 | 1.196 |
| 35% | 3.508 | 2.644 | 1.153 | 1.198 |
| 40% | 3.508 | 2.645 | 1.156 | 1.201 |

Sharpe barely changes with sizing; the original cumulative returns and
drawdowns are unchanged in all 90 rows. The 183 full daily intervals run from
March 4 00:00 UTC through September 3 00:00 UTC. The original cumulative return
still includes its partial first/last day, so its date bounds differ slightly.

Historical higher-cost execution stress:

| Per Position | 5m Delay Return | 5m Delay DD | 10m Delay Return | 10m Delay DD |
| --- | ---: | ---: | ---: | ---: |
| 20% | +13.46% | -18.70% | +14.52% | -21.75% |
| 25% | +16.60% | -22.91% | +17.93% | -26.43% |
| 30% | +19.62% | -26.94% | +21.22% | -30.83% |
| 35% | +22.51% | -30.80% | +24.38% | -34.98% |
| 40% | +25.26% | -34.54% | +27.40% | -38.90% |

For example, 40% weight does not justify planning around only a 17% drawdown:
the fixed 10m-delay scenario already approaches 39% on the same history. These
are stress scenarios, not estimates of the actual live delay distribution.
Historical drawdowns, including stress drawdowns, are not upper bounds on
future losses; five-minute-close risk also misses intrabar extremes.

## Covered Funding And Recent Stock Check

July-August shared book, higher costs, stock funding included:

| Per Position | No Extra Delay | 5m Delay | 10m Delay |
| --- | ---: | ---: | ---: |
| 20% | +9.73% | -0.58% | -1.95% |
| 25% | +12.01% | -0.97% | -2.66% |
| 30% | +14.22% | -1.44% | -3.47% |
| 35% | +16.34% | -2.00% | -4.36% |
| 40% | +18.39% | -2.65% | -5.32% |

Raising weight makes the losing delay scenarios worse. A weight increase does
not repair their net expectancy or the crypto training/live feature mismatch.

September 1-25 stock-only, higher costs, funding included, no extra delay:

| Per Position | Return | Marked DD | Daily Sharpe |
| --- | ---: | ---: | ---: |
| 20% | +0.40% | -8.60% | 0.370 |
| 25% | +0.46% | -10.64% | 0.373 |
| 30% | +0.49% | -12.65% | 0.376 |
| 35% | +0.52% | -14.61% | 0.379 |
| 40% | +0.52% | -16.53% | 0.382 |

This small recent comparator offers little extra net return for much larger
risk. It is not proof that larger sizing always underperforms, but is a useful
counterexample to extrapolating the strongest historical shared-book outcome.
These September Sharpes have only 25 daily returns, not a full-year estimate.

## Capital Usage And Interpretation

Amounts are entry notional as a fraction of realized equity, not a leverage
multiplier or a guaranteed percentage of the exchange's current free balance.
Different entry times and floating PnL mean two positions do not remain exactly
twice the target weight relative to current marked equity.

Across the tested windows/costs/delays, the minimum simple 1x headroom proxy was:

| Per Position | Minimum Proxy Headroom |
| --- | ---: |
| 20% | 55.79% |
| 25% | 45.66% |
| 30% | 35.85% |
| 35% | 26.37% |
| 40% | 15.66% |

The proxy is one minus gross position market value divided by simulated marked
equity. It uses completed candle closes, not exchange historical mark prices.
No negative proxy occurred, but that does NOT certify order acceptance,
maintenance margin, liquidation safety or available cash in the real account.
OKX distinguishes cross/isolated calculations and includes order loss in entry
requirements; account mode and fees matter. See the
[official margin rules](https://www.okx.com/en-sg/help/futures-margin-calculation-rules).

At 1x, a pair of stock positions both losing exactly 3% would cost approximately
1.2%, 1.5%, 1.8%, 2.1% or 2.4% of the sizing equity at the five respective
weights, before costs/gaps and ignoring differing entry equities. The crypto
stops are not fixed at the stock 3% value. Multi-trade drawdowns can be much
larger than this two-trade illustration.

## Decision Boundary

20% is a risk budget, not evidence of inefficient capital. Under this particular
historical delay test, 25% already has a 26.43% drawdown; 30% exceeds 30%, and
35-40% reaches about 35-39%. The appropriate live weight cannot be selected
without a loss budget, reliable live execution measurements and a decision
about the unresolved crypto features.

25% is the smallest requested incremental research candidate, not an automatic
production recommendation. There is insufficient evidence to call 35-40% safe
or superior merely because their retrospective returns are larger. None of the
weights passed a requirement that every covered funding/delay case be profitable.

This comparison assumes the shared capacity is two. The current source policy
still declares five shared slots and 20%; these results must not be implemented
by changing only the weight while leaving the five-slot policy in place.

## Artifacts And Validation

`results/shared_two_slot_weights_20260928/` contains:

- `summary.csv`: all 90 scenario/delay/cost/weight rows.
- `monthly.csv`, `components.csv`: monthly outcomes and sleeve contributions.
- `trades_*.csv`, `daily_*.csv`: resized trade ledgers and daily marked equity.
- `daily_returns_*.csv`: exact full UTC midnight-to-midnight returns for Sharpe;
  partial days are omitted and idle days retained. Do not substitute the older
  daily last-observation files, which can contain partial endpoint days.
- `capital_*.csv.gz`: full five-minute equity and exposure/headroom diagnostics.
- `controls.csv`: nine 20% higher-cost books reproduced, notionals and net returns.
- `manifest.json`: frozen input/code hashes, method and limitations.

The no-delay historical 20% controls also reproduce both previously published
cost-pair returns and marked drawdowns. After the Sharpe/September extension,
the full suite passes 228 tests. Two existing third-party
calendar deprecation warnings remain; `git diff --check` passes.

Sharpe is arithmetic mean daily return divided by sample daily standard
deviation, times sqrt(365), with zero risk-free return. Zero-variance/insufficient
samples are undefined, not infinity or manufactured zero. The addendum protocol
and code are recorded in the regenerated manifest; the original protocol was
not rewritten. No autocorrelation adjustment or statistical significance claim
is made. The July-August cases contain 62 complete daily returns.

Reproduce with the frozen local data:

```bash
.venv/bin/python -m scripts.analysis.shared_two_slot_weights
.venv/bin/python -m pytest -q
git diff --check
```

Remaining model limits: known retrospective samples; frozen Binance crypto
outcomes rather than current live inputs; partial funding; realized-equity
sizing defers cost/funding until trade close; fractional lots; no exchange
margin rejection, liquidation or size-dependent impact. All percentages should
be read with these limits, not as forecasts or maximum possible losses.
