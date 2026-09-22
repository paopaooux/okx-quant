# OKX proxy fallback research (2026-09-22)

## Result

Port 7893 is technically usable as a backup for port 7890, but it is not yet
usable by the configured trading API key. OKX rejected all three read-only
account probes through 7893 with HTTP 401 / code 50110, explicitly identifying
156.248.10.79 as an IP missing from the API key whitelist. Preserve the existing
allowed IPs and add the verified backup exit IP before enabling trading fallback.
This observed IP is not proof that the provider guarantees a static exit IP.

No production configuration, account setting, leverage, or order was changed.

## Direct probes

Requests used the repository's account client with an explicit per-session proxy
and environment proxy inheritance disabled. No order or leverage calls were made.

| Proxy | Public time API, three requests | Account config / positions / balance |
| --- | --- | --- |
| 7890 | 0.229 / 0.120 / 0.134 seconds; HTTP 200, code 0 | All succeeded |
| 7893 | 4.009 / 0.996 / 1.100 seconds; HTTP 200, code 0 | All rejected: HTTP 401, code 50110 |

These are short samples, not a long-term availability or latency guarantee.
Public health checks alone cannot detect the API key whitelist problem.

## Isolated failover experiment

Used the installed Mihomo v1.19.30 in a temporary directory with separate
loopback listener and controller ports. A test-only TCP relay forwarded the
primary candidate to 7890; the second candidate used 7893. Existing services were
not stopped or reconfigured. The relay simulated a closed primary connection and
an eight-second response stall. Requests only targeted the public OKX time API.

The temporary fallback group used interval 3 seconds, timeout 5000 milliseconds,
lazy false, expected HTTP status 200, and primary-before-backup ordering.

- Healthy primary: selected primary; public API succeeded.
- Primary disconnected: selected backup after approximately 9.52 seconds;
  public API succeeded.
- Primary restored: selected primary after approximately 0.50 seconds;
  public API succeeded.
- Primary stalled: selected backup after approximately 8.02 seconds;
  public API succeeded.

Times depend on health-check scheduling and connection setup. They are not an
upper bound or a production SLA. Existing established connections are not
transparently migrated by changing the selected outbound.

## Proposed production configuration

Keep trading at 7890 and downloads at 7893. Add an OKX-specific fallback group
inside the 7890 service, selecting the existing Proxies group first and the local
7893 HTTP proxy second. Route only OKX traffic to this group. Do not route the
download service back to 7890: that would create a dependency loop.

Merge these entries into /root/clashctl/resources/mixin.yaml, preserving existing
entries. This is a proposal, not an applied configuration:

```yaml
proxies:
  append:
    - name: okx-backup-7893
      type: http
      server: 127.0.0.1
      port: 7893
proxy-groups:
  prepend:
    - name: OKX-Fallback
      type: fallback
      proxies:
        - Proxies
        - okx-backup-7893
      url: https://www.okx.com/api/v5/public/time
      expected-status: 200
      interval: 10
      timeout: 5000
      lazy: false
      max-failed-times: 2
rules:
  prepend:
    - DOMAIN-SUFFIX,okx.com,OKX-Fallback
```

The new domain rule must precede the existing okx.com -> Proxies rule. The local
clashctl merger supports proxies.append, proxy-groups.prepend and rules.prepend.
Persist changes in mixin.yaml as well as the generated runtime configuration so
a subscription refresh does not discard them. Validate the fully merged config
with mihomo -t before a controlled reload.

Native fallback reacts to health-check availability/timeouts. max-failed-times
triggers a forced health check; it is not an exact two-consecutive-probe policy.
Native fallback also does not implement a prolonged recovery hold-down. More
specific hysteresis would require additional control logic.

## Remaining conditions

1. Add the backup exit to the existing API key IP whitelist, then verify all
   read-only account probes succeed through 7893. Do not disable the whitelist.
2. Verify that the download provider keeps its exit IP stable. A node/server
   hostname does not itself identify the IP seen by OKX.
3. Confirm the fully merged group in isolation and apply it only after the
   account-access condition is satisfied.
4. Retain reconciliation for ambiguous order submissions. After a timeout,
   query the original client order ID through the working route; do not blindly
   resubmit the order with a different ID.

This topology protects against primary upstream failure, including exhausted
primary subscription traffic when the backup subscription remains available.
It does not protect against the host or the 7890 Mihomo process failing, both
subscriptions failing, or an exchange-wide outage. Exchange-side protective
orders are a separate remaining requirement.

## References

- https://wiki.metacubex.one/config/proxy-groups/fallback/
- https://wiki.metacubex.one/en/config/proxy-groups/
- https://wiki.metacubex.one/config/proxies/http/
- https://www.okx.com/docs-v5/
