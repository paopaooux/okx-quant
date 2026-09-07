# Live Combination Alignment, 2026-09-07

## Deployment

Policy: `original_combo_v1`. The `okx-quant` container was recreated at
2026-09-07 07:56:41 UTC. Its four execution/data-loop source files were checked
by SHA-256 against the tested workspace files. Startup configuration validation
passed, and the trading loop completed with `simulated_header=0`.
The first full-universe refresh completed at 08:00:41 UTC: all 167 instruments
refreshed, with no reported refresh error. Post-deployment account reconciliation
still showed the same five legacy positions and no pending entry orders.

OKX remains both the signal data source and the execution venue, as requested.
No concentration filter, additional stock entry filter, or signal-invalid exit
from the eight-way ablation was enabled. Model weights and input construction
were not changed. Runtime dependency versions are pinned to the prior live image.

## Rules For New Positions

| Rule | Value |
|---|---|
| Stock universe | All 167 instruments in the existing stock universe file |
| Stock direction | Revert the first off-hours displacement in each window |
| Stock trigger | 600bp absolute displacement |
| Stock stop | 300bp from actual entry fill |
| Stock fixed target | None |
| Stock deadline | Earlier of cash open +60 minutes or event +30 hours |
| Stock capacity | At most 3 open/pending stock positions |
| Stock daily limit | At most 2 submitted stock entries per UTC day |
| Shared capacity | At most 5 open/pending instruments |
| Position size | 20% of account equity less tracked unrealized PNL, rounded down to exchange lots |
| Crypto | Eight symbols, c/rolling-730, tail 0.01, unchanged model and OKX features |
| Crypto deadline | Signal-bar start +49 quarter-hour bars: next-open entry plus 48-bar horizon |
| Signal-invalid exit | Disabled |

Stock acquisition now covers the full universe, not just TECH, and requests at
least 18 days for the session anchor and trailing event statistics. Entry
detection uses completed, non-future bars, checks each instrument's freshness,
and includes the next cash open across weekends and holidays. Old events are
not entered on restart.

Exit management runs before feature calculation. Management-only position
records cannot become entry signals. API read failures do not mean an empty
account. Entry and exit intents have durable client order IDs; uncertain orders
are reconciled and reserve capacity instead of being resubmitted with new IDs.

## Existing Positions

Five actual short positions were verified with the OKX positions API before
deployment: SMCI, NVDA, CRWV, ONDS, and AMD. They were not force-closed or resized.
They are explicitly marked `legacy_preserved`, retaining their 600bp stop,
600bp target, and original 30-hour deadlines. Consequently, the current book is
in transition, not a retroactively reconstructed original-strategy portfolio.

| Legacy Instrument | Existing Deadline (UTC) |
|---|---|
| SMCI-USDT-SWAP | 2026-09-08 04:00 |
| NVDA-USDT-SWAP | 2026-09-08 04:30 |
| CRWV-USDT-SWAP | 2026-09-08 05:00 |
| ONDS-USDT-SWAP | 2026-09-08 05:00 |
| AMD-USDT-SWAP | 2026-09-08 12:15 |

Stops or targets may close these positions before their deadlines. They count
toward the new entry limits. The deployment day's stock quota is conservatively
marked consumed because the old state did not retain a reliable daily counter.
Future UTC dates start a new quota; restarting the process does not reset it.

## Missing Model Inputs

The live BTC snapshot at 2026-09-07 07:58:02 UTC had 16 missing values among the
54 model inputs:

```text
tbr_z_96
oi_chg_96, oi_z_96, oi_ret_corr_96
tt_pos, tt_pos_z_96, tt_pos_chg_16
tt_acc, tt_acc_z_96, tt_acc_chg_16
taker, taker_z_96, taker_chg_16, taker_mean_16
crowding, crowding_z_96
```

The current adapter supplies no top-trader account/position ratios or original
taker ratio series. It also uses half-volume proxies for candle taker volumes
and volume in place of trade count. Constant proxies and gaps in positioning
history affect derived features. This describes the current implementation,
not a claim that no OKX endpoint could supply any additional data.

Missing inputs remain NaN and use the model's learned missing-value branches.
That keeps prediction executable; it does not reconstruct the missing data or
establish equivalent predictions. High cross-venue price correlation does not
establish equivalent order flow, positioning, or tail-entry signals. Each live
signal now records its actual `missing_features` list for inspection.

## Verification And Limits

- 51 local tests passed, including live-loop exit-before-entry behavior,
  uncertainty handling, restart persistence, holiday session detection, and
  unchanged legacy exits.
- All 299 accepted trades from the frozen combination export pass the live
  capacity and daily-quota rules when replayed on their historical schedule.
  This is an admission compatibility check, not a new live-equivalent backtest.
- The historical backtest has two-stage sleeve/sharing selection using ideal
  fills; live admission uses actual positions and pending orders. Missing bars,
  latency, rejected or partial orders, and ambiguous order status can change
  later admissions. Submission reservations deliberately err toward fewer
  trades when a fill is uncertain.
- Historical stop levels assume barrier fills. Live exits use current OKX
  prices and market orders, with actual fees, slippage and funding. Local
  polling is not an exchange-native protective stop, and cycle/API delays
  remain an operational risk.
- Retaining OKX features is an explicit exception to the original Binance
  model input pipeline. The old +50.28% backtest is not a prediction or guarantee
  of the deployed strategy's returns.

## Recovery Assets

Prior live scripts, pre-migration state, and prior Compose configuration:
`data/deploy-backup-original-policy-i20WUy/`.

Prior image tag: `okx-quant:before-original-policy-i20wuy`.
Do not blindly restore the old state over a changing real account. Any recovery
must reconcile current orders/positions first; the old version also contains
the management-signal re-entry defect fixed by this deployment.
