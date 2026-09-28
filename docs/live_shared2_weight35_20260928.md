# Approved Live Allocation Change

User decision on 2026-09-28: two shared stock/crypto slots, 35% per new entry.
Policy version: `shared2_weight35_v2`, replacing `original_combo_v1`.

- Shared capacity is two, including pending entries, across both asset classes.
- Stock capacity is two; the stock daily limit remains two entries per UTC day.
- Each new entry targets 35% of the existing USDT sizing-equity definition.
  Available margin, fee buffer and downward contract-lot rounding still cap it.
  Two full targets total approximately 70%, not a guarantee of free margin.
- Leverage remains 1x isolated; no new universe, signal, stop or exit rules.
- Existing quantities, stop/target widths and deadlines are preserved. No
  automatic top-up, resizing or forced liquidation on policy migration.
- If existing positions/pending entries occupy two or more slots, new entries
  wait for capacity. Existing positions continue normal exit management.
- Python defaults derive capacity/weight from the policy; Docker Compose and
  `.env.example` agree. Old overrides fail the existing startup validation.

Initially this was a code/configuration-only change. At the user's subsequent
request, the image was rebuilt and the `okx-quant` container recreated on
2026-09-28 at 10:06:33 UTC (18:06:33 Asia/Shanghai). The new image digest is
`sha256:e312a3501ad0b974bd94aa94f5d922fd86d8a1979932d247fa9670802d316713`.

Preflight validated the policy and US/HK/Korea calendar imports before stopping
the old container. Startup logs confirm live mode, two shared slots, 0.35
weight and 1x isolated execution. All four supervised workers were running,
account sync completed, state migration persisted, and entry reconciliation
was successful. Restart count was zero at the post-deployment check.

The existing SKDD short remained size 2.0 with its original policy metadata
and September 28 14:30 UTC deadline; one protective-order record and no pending
entries were present. No manual order, top-up or position resize was sent.
The known crypto missing-feature issue remains outside this allocation change.

The earlier backtest reports describe their frozen research settings and are
not rewritten as evidence of deployment. Historical results and drawdowns are
not guarantees of future performance or maximum losses.
