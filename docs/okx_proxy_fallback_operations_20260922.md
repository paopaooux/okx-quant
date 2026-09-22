# OKX proxy failover

## Applied status

Applied on 2026-09-22. Both proxy services are active. The main service selects
FlowerCloud and the backup service selects OKX-SG07. Signed account config,
account-available SWAP instruments, and positions succeeded through both 7890
and 7893 after deployment. The trading container was not restarted (its start
time remains 2026-09-14T08:50:02Z).

The actual rollback archive is:
`/root/clashctl/backups/okx-fallback-20260922-162219/`.

| Isolated fault or recovery | Result | Observed selection delay |
| --- | --- | --- |
| Primary disconnected | Singapore07 | 6.03 s |
| Primary and Singapore disconnected | HongKong03 | 7.03 s |
| Singapore recovered, primary still down | Singapore07 | 7.53 s |
| Primary recovered | FlowerCloud | 5.02 s |
| Primary stalled for 8 s | Singapore07 | 14.06 s |
| Primary recovered again | FlowerCloud | 1.51 s |

All seven scenarios, including the initial healthy baseline, passed public and
signed read-only account checks. Faults were injected only into temporary proxy
relays, not into production. These observations are not timing guarantees.

## Topology

The requested preference order is FlowerCloud, then backup-provider Singapore07,
then backup-provider HongKong03. The priority is availability-based, not a race
to the lowest latency. The other tested nodes are not in the fallback pool.

```text
OKX requests -> 7890 -> OKX-Fallback
                         1. Proxies (existing FlowerCloud selection)
                         2. local HTTP 7893 -> OKX-Backup
                                                1. OKX-SG07
                                                2. OKX-HK03
```

Only the main service's okx.com domain-suffix rule changes. Other domains keep
their existing routing. Port 7893 continues to serve its existing download
clients, but replaces the unreliable old HongKong02 node with the two-node pool.
The backup service never routes to 7890, avoiding a routing cycle.

The main FlowerCloud selection was `Hong Kong Advanced IEPL Line 2` at setup
time. Its existing selector is retained. Changing `Proxies` manually also changes
the primary candidate; this is not an automatic search through all FlowerCloud
nodes. A broken primary node can therefore trigger the external backup even if
another FlowerCloud node is healthy.

## Health policy

Both groups use native Mihomo fallback with:

- URL: `https://www.okx.com/api/v5/public/time`
- Expected HTTP status: 200
- Check interval: 10 seconds; timeout: 5000 milliseconds; lazy: false
- Priority: first healthy member in the configured order
- max-failed-times: 2 (triggers health checking; not exactly two failed probes)

The primary continues to be checked while a backup is selected. A successful
health check allows automatic return to the higher-priority member. There is no
additional consecutive-success or recovery hold-down timer in this native mode.
Detection time depends on check scheduling and both layers; it is not a fixed
10-second guarantee. Established TCP connections are not forcibly terminated or
migrated on a selection change. In-flight requests may still fail or time out.

## Persistent files

- `/root/clashctl/resources/mixin.yaml`: durable main routing override
- `/root/clashctl/resources/runtime.yaml`: merged active main configuration
- `/root/clashctl/okx-download/config.json`: dedicated backup configuration
- `deploy/okx-proxy-fallback.mixin.yaml`: credential-free main override reference

The main config is generated using the installed clashctl merger. The generated
result is checked for unrelated changes, then validated with `mihomo -t`.
Credentials are taken from the existing local subscription, not committed here.
The backup controller listens only on 127.0.0.1:9093 and requires a random secret.
The backup nodes are pinned copies: refreshing the source subscription does not
automatically update this standalone service's credentials or server addresses.

## Deployment and rollback

Before applying, archive all three live files under
`/root/clashctl/backups/okx-fallback-<timestamp>/` with root-only access.
Replace and restart the dedicated backup service first, verify signed read-only
account access through 7893, then atomically install the main mixin/runtime and
hot-reload Mihomo using `PUT /configs?force=false`. Verify signed read-only
account access through 7890 and inspect both group selections afterward.
Do not rebuild or restart the trading container for this proxy-only change.

On deployment failure, restore the archived files, restart
`mihomo-okx-download.service`, and reload the main original runtime. A later
rollback must first check for intervening user changes; do not blindly overwrite
newer configuration. Archives contain secrets and must not be published.

## Verification and limits

An isolated three-process test uses temporary relay faults, not live service
disruption. It tests primary failure, Singapore failure, Hong Kong fallback,
Singapore recovery, primary recovery, a stalled primary, and recovery again.
Each scenario checks public time, signed account configuration, account-available
SWAP instruments, and positions. No orders or leverage changes are submitted.

The selected backup nodes passed read-only account checks after the user added
their observed IPv4 exits to the whitelist:

- Singapore07 (`sg3.vpnmiao.com`): 185.190.58.58
- HongKong03 (`hk3.vpnmiao.com`): 89.185.26.139

These are observed addresses, not a provider guarantee of immutable IPv4 exits.
Public health checks do not detect whitelist, API permission, account, or
regional restrictions. This configuration does not bypass those restrictions or
certify eligibility to trade derivatives. In particular, HTTP success from a
public endpoint and read-only account access do not prove order acceptance.

An ambiguous order timeout must be reconciled by its original client order ID,
not blindly resubmitted on another route. Proxy changes do not deploy any pending
trading-code fixes or exchange-side protective orders. The topology cannot cover
failure of the host/main proxy process, both providers, or the exchange itself.
Both backup nodes share the same subscription quota.

## References

- https://wiki.metacubex.one/config/proxy-groups/fallback/
- https://wiki.metacubex.one/config/proxy-groups/
- https://www.okx.com/docs-v5/en/
