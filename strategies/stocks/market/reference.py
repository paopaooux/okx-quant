"""Reviewed listing-market routing for cash-session reversion signals.

US listings verified against Nasdaq directories dated 2026-09-25. A fund or
ADR follows its own listing venue, not the country of its portfolio/company.
Unknown and Pre-IPO instruments have no implicit US fallback. See docs/stock_reference_markets.md.
"""
from functools import lru_cache

import exchange_calendars as xcals
import numpy as np
import pandas as pd

VERSION = "reference_sessions_v1_20260928"
US_LISTINGS = frozenset("""
AAOI AAPL ADBE AEHR ALAB AMAT AMD AMZN APLD APP ARM ASML ASTS AVGO AXTI BB BE
BMNR BOT BRKB BSP BX CBRS CGNX CIEN COHR COIN COST CRCL CRDO CRM CRWD CRWV CSCO
DDOG DELL DKNG DRAM EWJ EWT EWY EWZ FLNC FLY FWDI GEV GLW GME GOOGL HIMS HOOD
HPE IBM INFQ INTC INTW IONQ IREN ISRG IWM JNJ KLAC KO KORU LITE LLY LRCX LUNR
LYTE MARA META MRK MRNA MRVL MSFT MSTR MU MUU MVLL NBIS NET NFLX NOK NOW NVDA
NVDL OKTA ON ONDS ORCL OSCR OUST PENG PLTR POET PURR PYPL QCOM QNT QQQ RAM RDDT
RDW RIOT RIVN RKLB ROK SHAZ SHLD SHOP SIMO SKDD SKHY SKUU SMCI SMH SNDK SNOW
SNXX SONY SOXL SOXS SPCH SPCX SPY SQQQ STRC TER TMF TQQQ TSEM TSLA TSLL TSM
TTMI TTWO TWLO UNH URNM USAR USO UVXY VRT WDC WEN WMT XBI XLE ZM
""".split())
MARKETS = {ticker: "XNYS" for ticker in US_LISTINGS}
MARKETS.update({ticker: "XHKG" for ticker in (
    "MINIMAX", "ZHIPU", "POPMART", "XIAOMI", "CSOPSKHYNIX2L", "CSOPSAMSUNG2L")})
MARKETS.update({ticker: "XTKS" for ticker in ("KIOXIA", "SOFTBANK")})
MARKETS.update({ticker: "XKRX" for ticker in ("SAMSUNG", "SKHYNIX", "HYUNDAI")})


def reference_market(inst_id):
    return MARKETS.get(str(inst_id).removesuffix("-USDT-SWAP"))


@lru_cache(maxsize=4)
def calendar(market):
    if market not in {"XNYS", "XHKG", "XTKS", "XKRX"}:
        raise ValueError(f"Unsupported reference market: {market}")
    return xcals.get_calendar(market)


def closed_windows(market, start, end):
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    schedule = calendar(market).schedule
    if start.date() < schedule.index[0].date() or end.date() > schedule.index[-1].date():
        raise ValueError(f"Reference calendar coverage unavailable: {market}")
    schedule = schedule.loc[(start-pd.Timedelta(days=10)).date().isoformat():
                            (end+pd.Timedelta(days=10)).date().isoformat()]
    rows = []
    for i in range(1, len(schedule)):
        previous, current = schedule.iloc[i-1], schedule.iloc[i]
        close, opening = previous["close"], current["open"]
        if opening < start or close > end:
            continue
        gap = (schedule.index[i]-schedule.index[i-1]).days
        kind = "overnight" if gap == 1 else (
            "weekend" if gap == 3 and schedule.index[i].weekday() == 0 else "holiday")
        rows.append(dict(close_ts=close, open_ts=opening,
                         prev_day=schedule.index[i-1].date(), next_day=schedule.index[i].date(),
                         hours=(opening-close).total_seconds()/3600, kind=kind))
    return pd.DataFrame(rows, columns=["close_ts", "open_ts", "prev_day", "next_day", "hours", "kind"])


def market_state(index, market):
    """Completed-bar classification; lunch is closed but not an overnight event."""
    schedule = calendar(market).schedule
    stamps = pd.DatetimeIndex(index).as_unit("ns").asi8
    opens = pd.DatetimeIndex(schedule["open"]).as_unit("ns").asi8
    closes = pd.DatetimeIndex(schedule["close"]).as_unit("ns").asi8
    if len(stamps) and (stamps.min() < opens[0] or stamps.max() > closes[-1]):
        raise ValueError(f"Reference calendar coverage unavailable: {market}")
    locations = np.searchsorted(opens, stamps, side="right")-1
    safe = np.maximum(locations, 0)
    active = (locations >= 0) & (stamps <= closes[safe])
    breaks = pd.DatetimeIndex(schedule["break_start"]).as_unit("ns").asi8[safe]
    resumes = pd.DatetimeIndex(schedule["break_end"]).as_unit("ns").asi8[safe]
    active &= ~((stamps >= breaks) & (stamps < resumes))
    return pd.Series(np.where(active, "cash", "closed"), index=index)


def stock_events(frames, config, *, now=None):
    """Shared live/research router. A failed market produces no new entries."""
    from strategies.stocks.research.events import off_hours_dislocation
    groups = {}
    for inst, frame in frames.items():
        market = reference_market(inst)
        if market and len(frame):
            groups.setdefault(market, {})[inst] = frame
    results = []
    for market, group in groups.items():
        start = now-pd.Timedelta(days=8) if now is not None else min(f.index.min() for f in group.values())
        end = now+pd.Timedelta(days=8) if now is not None else max(f.index.max() for f in group.values())
        try:
            windows = closed_windows(market, start, end)
            if now is not None:
                windows = windows[(windows.close_ts < now) & (windows.open_ts > now)]
                group = {inst: f.loc[f.index <= now] for inst, f in group.items()}
            found = off_hours_dislocation(group, config, windows=windows, reference_market=market)
        except (ValueError, KeyError) as exc:
            if now is None:
                raise  # Research must not silently compare incomplete universes.
            print(f"STOCK CALENDAR SKIP {market}: {type(exc).__name__}", flush=True)
            continue
        if len(found):
            found["reference_market"] = market
            found["calendar_policy_version"] = VERSION
            results.append(found)
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()
