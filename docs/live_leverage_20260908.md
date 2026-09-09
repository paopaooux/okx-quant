# Live 1x leverage, 2026-09-08

The user requested live execution without added leverage, matching the shared
five-slot backtest. `scripts/combinations/run.py::_pooled_curve` budgets
`equity / 5` per accepted entry and books `notional * net` at exit without a
leverage multiplier. This is an entry-sizing rule, not continuous rebalancing
to a strict gross-exposure cap as equity and market prices change.

At 2026-09-08 11:22:42 UTC, the real OKX account confirmed both open isolated
swap positions at 1x after explicit set-leverage requests and leverage-info
readback:

| Instrument | Contracts | Previous leverage | Verified leverage | Margin (USDT) |
|---|---:|---:|---:|---:|
| ANTHROPIC-USDT-SWAP | -0.069 | 2 | 1 | 14.5659 |
| AMD-USDT-SWAP | -0.02 | 3 | 1 | 9.6884 |

Contract quantities were unchanged. Margin was increased by OKX. Nominal
exposure remains sized to 20% at entry, rounded down to exchange lots; AMD's
0.01 contract lot explains its smaller allocation. Lowering exchange leverage
changes collateral requirements, not the PNL per price move for fixed size.

`DemoClient.ensure_unleveraged` now reads isolated leverage, sets it to 1 when
needed, and verifies it with a separate read. New entries require this check
before quota or order-intent reservation. Open strategy positions are checked
every loop after exit management; a repair failure blocks new entries while
preserving exit management on subsequent loops. Observe mode makes no leverage
changes. No account-wide settings for unused instruments were modified; they
are checked before each future entry.

Deployment used the previously running image plus only `auto_demo.py` and
`okx_demo.py`, preserving its dependencies and other source files. Recovery
image: `okx-quant:before-1x-20260908`. The container was recreated and both
deployed files' SHA-256 hashes matched the workspace. Startup reported
`leverage=1 margin_mode=isolated simulated_header=0`.

Validation: 59 tests passed, including independent leverage readback failure,
entry blocking without quota consumption, exit-before-repair ordering, and
observe-mode behavior. Exchange positions were checked again after deployment
and both remained at 1x with their original quantities.
