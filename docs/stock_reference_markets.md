# Stock Reference-Market Calendars

Version: `reference_sessions_v1_20260928`. This changes new stock signals, not the
exit policy of an already-open position. Do not deploy without reviewing the
paired study in `results/reference_calendar_study_20260928`.

## Routing

`strategies/stocks/market/reference.py` contains an explicit reviewed mapping.
No ticker prefix, company nationality or unknown-symbol fallback selects a
market. New/unreviewed instruments are ineligible until the map is reviewed.

| Calendar | Instruments |
| --- | --- |
| XNYS | Reviewed US listings, including US-listed ADRs, ETFs and STRC preferred stock |
| XHKG | MINIMAX, ZHIPU, POPMART, XIAOMI, CSOPSKHYNIX2L, CSOPSAMSUNG2L |
| XTKS | KIOXIA, SOFTBANK |
| XKRX | SAMSUNG, SKHYNIX, HYUNDAI |

For the current 167-instrument feed, 160 are mapped. ANTHROPIC, CXMT, KR200,
MOONSHOT, OPENAI, SHEIN and UNITREE remain disabled for new signals: no reviewed
cash-listing market is assigned. This is not a claim about their permanent IPO
status. Review contract/index identity and listing transitions before enabling.

KORU, SKDD, SKUU and the SKHY ADR use the US listing calendar, even though their
economic exposure is Korean. The two CSOP products use their Hong Kong product
listing calendar, not their Korean component market. TSM/SONY US depositary
receipts similarly retain US sessions. This market fix does not separately
exclude leveraged products or validate the reversion hypothesis for them.

## Rules

- Pinned `exchange-calendars==4.13.2` supplies sessions, holidays, early closes,
  timezone/DST changes and lunch-break information. A calendar coverage error
  suppresses entries for that market; it never falls back to US hours.
- Signals use the gap between consecutive full sessions' close and next open.
  Lunch breaks are excluded from cash-volume diagnostics but do not become
  overnight reversion windows. The existing minimum eight-hour closure and
  minimum one-hour time-to-open tests remain unchanged.
- Anchors are the perpetual's own known close at the reference-market closing
  timestamp, not independently observed underlying-stock quotes. A 6% change is
  therefore not a measured 6% premium to the underlying.
- New deadlines remain `min(reference_open + 60 minutes, event + 30 hours)`.
  Trigger 6%, stop 3%, no fixed take-profit, slot limits and daily quota unchanged.
- Cached signals and order admission reject a reference open that has passed.
  A second check after account/leverage/quote calls protects the submission path.
- Positions persist their reference market, calendar policy version and anchor
  timestamp. Existing positions keep their stored deadline and protective stop.
  Unknown-market positions are still managed; only new signals are excluded.

## Backtests

Canonical `scripts/stocks/research_offhours_candidates.py` now defaults to
`--calendar-mode reference`. `--calendar-mode legacy-us` explicitly reproduces
the old convention. Historical diagnostic modules retain their old baselines;
use `scripts/analysis/reference_calendar_study.py` for the paired correction.

The paired study uses frozen close-time-indexed candles from September 11-25,
2026. Three comparisons separate unknown-market exclusion from calendar changes:
all instruments on US hours, mapped instruments on US hours, mapped instruments
on their own hours. Each reruns signals and admissions, including replacement
trades. It is stock-only, with 10/30/44/68 bp assumed round-trip costs, no measured
funding or live delays. Historical stop-barrier assumptions are retained.

The older six-month candidate archive cannot reconstruct different-market
signals without old OHLC candles. Do not relabel the earlier US-only ablation as
a six-month test of this fix. The mapping is current, not a historical security
master: past listing conversions still require point-in-time validation.

## Sources and Maintenance

- [Calendar library](https://github.com/gerrymanoim/exchange_calendars)
- [Nasdaq listings](https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt)
  and [other US listings](https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt),
  dated September 25, 2026 and archived in the US-only study.
- [MiniMax HKEX filing](https://www1.hkexnews.hk/listedco/listconews/sehk/2026/0313/2026031301613.pdf)
- [Xiaomi listing](https://ir.mi.com/corporate-information/company-profile)
- [Pop Mart HKEX filing](https://www.hkexnews.hk/listedco/listconews/sehk/2025/0715/2025071500213.pdf)
- [CSOP Samsung listing](https://csop.onlineminisite.com/samsunglandi/en/)
  and [CSOP SK Hynix](https://csop.onlineminisite.com/skhynixleveraged/en/)
- [Kioxia listing](https://www.kioxia-holdings.com/ja-jp/ir.html)
  and [SoftBank Group listing](https://group.softbank/en/ir/stock/info)
- [Samsung stock information](https://org-sec-b2c.samsung.com/sec/ir/ir-resources/faq/)
  and [Hyundai stock information](https://www.hyundai.com/worldwide/en/company/ir/stock-information/stock-information)

Calendar libraries cannot predict emergency exchange closures. Recheck pinned
calendar data, IPO/conversion events and OKX reference-instrument specifications
before deploying or expanding the universe. The mapping is listing-market
routing, not a complete reconstruction of all OKX index pricing components.
