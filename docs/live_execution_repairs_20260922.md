# Live execution repairs (2026-09-22)

## Scope and deployment status

The first five findings from the live execution review were deployed on
2026-09-22 at 08:32:44 UTC (16:32:44 Asia/Shanghai). The model and strategy
thresholds were not changed. No test trades were sent; the resumed strategy
created its normal protective stop for the existing position.

## Deployment verification

- All 109 tests passed before rebuilding. A non-trading container also passed
  module imports and strategy configuration validation with EXIT_INTERVAL=5.
- Old image retained as `okx-quant:pre-bugfix-20260922`.
- State and database backed up after stopping the old container, under
  `data/deploy_backups/bugfix-20260922T083230Z/`; SQLite integrity check passed.
- New image: `sha256:77237698f42b6dd86dd0ed9b3c8c94f2e8d5dde92f31edb7a539ad2b09e28e06`.
  Running execution/client/risk module hashes match the workspace.
- Supervisor and all four children are running; container restart count is zero
  at verification. Real trading remains enabled with the original configuration.
- Existing CSOPSKHYNIX2L-USDT-SWAP long position (3 contracts) was retained.
  Its original 3% stop produced a live exchange conditional order at 5.246,
  with closeFraction=1 and reduceOnly=true, confirmed by exchange readback.
- Account sync completed the initial backfill (115 fills, 140 bills) and another
  incremental cycle. Position snapshots continued advancing after startup.
- FlowerCloud -> Singapore07 -> HongKong03 proxy failover was already deployed
  separately; this container replacement did not alter its configuration.

## Behavior

1. The API client checks both top-level code and per-order sCode. Business
   rejection releases an entry intent and its unfilled stock quota, or clears a
   rejected exit intent, with exponential retry delays from 5 to 300 seconds.
   Network errors, internal timeouts and duplicate IDs preserve identity. New
   regular orders carry a persisted 10-second expTime. Lost acknowledgements can
   recover only after expiry plus 60 seconds, three negative order lookups and
   successful pending-order/position checks. Entry recovery requires no exposure;
   exit recovery uses the currently confirmed position size and reduceOnly.
   Legacy uncertain intents without expiry remain reserved for manual review.

2. One execution writer owns the state file and an exclusive process lock.
   Signal computation runs in a separate worker with its own HTTP session. The
   management loop targets 5 seconds (AUTO_EXIT_INTERVAL); slow network calls can
   still extend that interval. Deadline exits do not require a ticker. If the
   positions endpoint fails, known local positions can still be reduced while
   new entries are blocked. The state file and containing directory are fsynced.

   Filled positions receive a conditional market stop using closeFraction=1,
   reduceOnly=true, isolated margin and net position mode. Trigger prices follow
   each position's existing width and last-price basis, rounded inward to tick
   size. The algo client ID is persisted before submission, and live state is
   verified by readback. Failed protection triggers a reduce-only exit attempt
   and blocks further entries. Orphan stops are canceled and retained in state
   until cancellation is confirmed. Outstanding exchange orders block re-entry.
   Restored legacy positions lacking a stop width block new entries and retain
   their existing deadline; the code does not invent a new strategy stop width.

3. Sizing uses settled USDT cash and available USDT funds, accounting for active
   positions and pending entries. It reserves 1% by default
   (AUTO_CAPITAL_BUFFER, minimum 0.5%). Each lot rounds down within both its 20%
   slot and remaining budget. Available funds are refreshed before each entry;
   local reservations also cap successive orders if exchange balances lag.
   Missing contract/balance metadata blocks dynamic entry sizing.

4. Each crypto instrument checks its own confirmed candles, metrics and final
   feature row. OI and account-ratio source times are tracked separately. The
   existing one-bar metrics lag in build.py is accounted for explicitly when
   validating aligned features. BTC-dependent features require fresh BTC input.
   Signals are checked again at admission, including stock event age. One stale
   symbol or unavailable entry ticker does not suppress unrelated fresh entries.

5. The supervisor now starts an independent account-sync process. The fill and
   bill streams page the three-month history APIs independently, committing each
   page and its cursor in one SQLite transaction. Restarts resume the active
   bounded window. A completed window advances the watermark, with one hour of
   overlap on subsequent polls. Backfill starts 89 days ago; older unavailable
   history requires a separate exchange export. Default interval is 60 seconds.
   Fill identities include the instrument, and replays update legacy timestamps
   from exchange execution times. Statistics use fills and both funding-fee
   directions rather than the old manual-orders table. Account totals are not
   automatically strategy-attributed performance.

## Validation

- Automated coverage includes rejected/ambiguous orders, backoff and quota
  recovery, partial-entry cancellation, deadline exits without tickers, account
  lookup failure, blocked signal computation, stop readback and restart identity,
  orphan cancellation, budget caps, per-symbol stale data and ledger recovery.
- Real read-only history requests retrieved 115 fills and 139 bills into
  /tmp/okx-account-sync-validation-20260922.sqlite3. Both streams completed their
  bounded backfill. Production account data was not changed by this validation.
- The updated crypto signal builder successfully built signals for all eight
  instruments through the existing primary proxy, with metrics timestamps.
- Pre-deployment stop placement tests used simulated API responses. Deployment
  subsequently verified the normal strategy-created live stop described above.

## Activation

Before activation, preserve the current image and back up the state and SQLite
database. Stop the old execution process before starting the new image: the old
version does not participate in the new state-file lock. Rebuild and replace only
the trading service, preserving data mounts and proxy configuration. Verify the
account-sync process, its stream watermarks, and any live stop readbacks.

Once the new version creates protective orders, reverting to the old image is
not sufficient rollback: reconcile/cancel its owned protective orders and order
intents before letting the old version trade. Never delete uncertain intents to
force a restart. Network failover is a separate deployed change; see
`docs/okx_proxy_fallback_operations_20260922.md`.

References: https://www.okx.com/docs-v5/en/ and
https://www.okx.com/docs-v5/log_en/ (full-position stop order parameters).
