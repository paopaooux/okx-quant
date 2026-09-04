# Crypto Perpetual Direction Strategy: rolling-730 / tail-0.01

## Status

研究候选，允许纸面交易和小规模成交验证；尚未达到自动实盘门槛。

## Fixed specification

- Universe: `BTCUSDT`, `ETHUSDT`, `SOLUSDT`, `BNBUSDT`, `XRPUSDT`, `DOGEUSDT`, `ADAUSDT`, `LINKUSDT`.
- Bar: 15 minutes. Entry is the next bar open after the signal bar.
- Features: Binance USDT-M 15m OHLCV plus lagged 5m derivatives metrics.
- Label: configuration `c`, symmetric triple barrier, `K=2.0`, horizon `H=48` bars.
- Model: direction-pure LightGBM, six chronological folds, purge/embargo, trailing 730-day training window.
- Signal: each fold's training score quantiles; `p_up` above the 99th percentile is long, below the 1st percentile is short. Thresholds are never computed from test scores.
- Live portfolio: one shared five-slot pool across stock and crypto `-USDT-SWAP` contracts in OKX one-way/net mode; each accepted slot targets 20% of account equity. Each instrument has one net position, so an opposite signal must close the current position before reversal.
- Costs: report at 10bp round trip and stress at 16bp; funding is audited separately before live use.

## Dynamic-universe research branch

Dynamic inclusion is permitted only as a forward, fold-local decision. For each
walk-forward fold, score symbols on the validation tail that was not used to fit
the LightGBM trees; require at least 20 validation signals, rank by
`n/(n+50) * mean_net`, and freeze at most six eligible symbols for the following
test block. The test block reads only the frozen `eligible` flag. It never uses
test outcomes or the complete OOS result to choose symbols.

The current dynamic branch (10bp) is a lower-return, lower-drawdown research
variant: CAGR 24.7%, daily Sharpe 1.59, max drawdown -7.3% across 696 trades,
versus the frozen eight-symbol candidate at 39.7% / 2.01 / -12.5%. Keep the
frozen eight-symbol universe as the primary paper-trading candidate until a new
universe rule is validated on a separately frozen forward period.

Capacity is a separate control from the universe. The slot sweep keeps the
eight-symbol pool; the standalone crypto sleeve uses 1/8 weights, while the
live stock+crypto shared pool uses 1/5 weights. Fixed 3/4/5/8
concurrent caps and a range-adaptive 3/5/7 cap are reported with stock-aligned
drawdown-recovery and holding-time fields.

Long and short are separate signal directions, but live execution is one-way:
an instrument cannot hold long and short simultaneously. A reverse signal is
held until the current net position exits, subject only to the shared slot/risk
budget; there is no forced 50/50 long-short allocation.

The primary slot results use the LightGBM trained on the full frozen eight-symbol
universe.

## Evidence

The broad 8-symbol OOS sample covers 2023-04-11 through 2026-09-03. The secondary split uses data before 2025-01-01 for selection and freezes the choice for the later validation period.

| segment | trades | CAGR | daily Sharpe | max drawdown |
| --- | ---: | ---: | ---: | ---: |
| selection (pre-2025) | 481 | 38.8% | 2.61 | -4.0% |
| frozen validation (2025 onward) | 667 | 36.5% | 1.73 | -10.2% |
| full OOS, 10bp | 1,148 | 39.7% | 2.01 | -12.5% |
| full OOS, 16bp | 1,148 | 36.0% | 1.87 | -12.9% |

At 10bp, calendar-year Sharpe was positive in 2023, 2024, 2025 and 2026-to-date; the weakest year was 2026-to-date at 1.43. Reversing the signal and random selection were strongly negative, which is a required control rather than an optional comparison.

## Reproduction

```bash
cd /root/program/okx-quant
.venv/bin/python -m scripts.data.fetch_binance \
  --start=2023-04-01 \
  --symbols=BNBUSDT,XRPUSDT,DOGEUSDT,ADAUSDT,LINKUSDT
.venv/bin/python -m scripts.data.build \
  --symbols=BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,DOGEUSDT,ADAUSDT,LINKUSDT \
  --output=data/panel_broad.csv.gz
.venv/bin/python -m scripts.modeling.train_dir c \
  --roll=730 --panel=data/panel_broad.csv.gz
# optional forward-only dynamic-universe research branch
.venv/bin/python -m scripts.modeling.train_dir c \
  --roll=730 --panel=data/panel_broad.csv.gz \
  --dynamic-universe --dynamic-prior-n=50
```

The current audit artifacts are `results/crypto/oos_dir_c_roll730.csv.gz` and the model fold files under `models/`. Binance history is used for features and labels; OKX execution costs are a stress assumption, so live deployment still requires OKX fill, spread, depth and funding measurements.

## Promotion gate

Keep paper-only until at least one additional untouched forward quarter is positive after measured fees, slippage and funding, with no single symbol responsible for the result. Do not increase leverage to turn the annualized extrapolation into a target.
