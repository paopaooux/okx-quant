"""Continuously refresh SEC 8-K filings and OKX tokenized-stock 5m candles.

This process has no trading credentials and never places orders. It maintains
the local files consumed by ``auto_demo`` so a stale container restart cannot
turn an old snapshot into a live signal.
"""
from __future__ import annotations

import json
import os
import time
import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from strategies.stocks.market.okx import OKXClient, OKXError, update_cache

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("AUTO_STOCK_DATA", ROOT / "data" / "stocks_swap"))
FILINGS = Path(os.environ.get("AUTO_STOCK_FILINGS", DATA / "sec_filings_raw.csv"))
UNIVERSE = Path(os.environ.get("AUTO_STOCK_UNIVERSE", DATA / "universe.csv"))
INTERVAL = max(60, int(os.environ.get("STOCK_REFRESH_INTERVAL", "300")))
SEC_INTERVAL = max(30, int(os.environ.get("STOCK_SEC_INTERVAL", "60")))
CANDLE_INTERVAL = max(60, int(os.environ.get("STOCK_CANDLE_INTERVAL", str(INTERVAL))))
HISTORY_DAYS = max(2, int(os.environ.get("STOCK_HISTORY_DAYS", "4")))
SEC_LOOKBACK_DAYS = max(7, int(os.environ.get("SEC_LOOKBACK_DAYS", "30")))
SEC_UA = os.environ.get("SEC_USER_AGENT", "okx-quant research contact@example.com")
STATUS = Path(os.environ.get("AUTO_STOCK_STATUS", DATA / "data_status.json"))


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": SEC_UA, "Accept-Encoding": "gzip, deflate"})
    proxy = (os.environ.get("SEC_PROXY_URL") or os.environ.get("OKX_PROXY_URL")
             or os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY"))
    if proxy:
        s.proxies.update({"http": proxy, "https": proxy})
    return s


def refresh_sec() -> int:
    if not UNIVERSE.exists():
        return 0
    universe = pd.read_csv(UNIVERSE)
    # The migrated universe intentionally contains exchange metadata only;
    # recover CIKs from the archived SEC file and keep that mapping durable.
    old = pd.read_csv(FILINGS) if FILINGS.exists() else pd.DataFrame()
    cik_by_ticker = {}
    if not old.empty and {"ticker", "cik"}.issubset(old.columns):
        cik_by_ticker = old.dropna(subset=["ticker", "cik"]).drop_duplicates("ticker").set_index("ticker")["cik"].to_dict()
    rows = []
    session = _session()
    attempted = failures = 0
    records = universe.copy()
    records["ticker"] = records["baseCcy"].astype(str).str.replace("^X", "", regex=True)
    records["cik"] = records["ticker"].map(cik_by_ticker)
    for record in records.dropna(subset=["cik", "ticker", "instId"]).drop_duplicates("cik").itertuples(index=False):
        attempted += 1
        cik = str(record.cik).split(".")[0].zfill(10)
        url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        try:
            response = session.get(url, timeout=20)
            response.raise_for_status()
            recent = (response.json().get("filings") or {}).get("recent") or {}
            n = len(recent.get("accessionNumber", []))
            ticker = str(record.ticker)
            inst = str(record.instId)
            for i in range(n):
                form = str(recent.get("form", [""] * n)[i])
                if form not in {"8-K", "8-K/A"}:
                    continue
                accepted = recent.get("acceptanceDateTime", [""] * n)[i]
                if not accepted:
                    continue
                ts = pd.to_datetime(accepted, utc=True, errors="coerce")
                if pd.isna(ts) or ts < pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=SEC_LOOKBACK_DAYS):
                    continue
                rows.append({
                    "instId": inst, "ticker": ticker, "cik": int(float(record.cik)),
                    "form": form, "accepted": ts.isoformat(),
                    "filing_date": recent.get("filingDate", [""] * n)[i],
                    "items": recent.get("items", [""] * n)[i] or "",
                    "accession": recent.get("accessionNumber", [""] * n)[i],
                    "doc_desc": recent.get("primaryDocument", [""] * n)[i],
                    "accepted_source": "sec_submissions",
                })
        except (requests.RequestException, ValueError, KeyError, IndexError) as exc:
            failures += 1
            print(f"SEC {record.ticker} refresh failed: {exc}", flush=True)
    if attempted and failures == attempted:
        raise RuntimeError(f"all {attempted} SEC submissions requests failed")
    if not rows:
        return 0
    fresh = pd.DataFrame(rows)
    known = set(old.get("accession", pd.Series(dtype=str)).dropna().astype(str)) if not old.empty else set()
    new_count = int((~fresh["accession"].astype(str).isin(known)).sum())
    merged = pd.concat([old, fresh], ignore_index=True, sort=False)
    merged = merged.drop_duplicates("accession", keep="last").sort_values("accepted")
    FILINGS.parent.mkdir(parents=True, exist_ok=True)
    tmp = FILINGS.with_suffix(".tmp")
    merged.to_csv(tmp, index=False)
    tmp.replace(FILINGS)
    return new_count


def refresh_candles(client: OKXClient) -> int:
    if not UNIVERSE.exists():
        return 0
    universe = pd.read_csv(UNIVERSE)
    count = 0
    attempted = failures = 0
    eligible = universe.loc[universe.get("kind", "single").astype(str) != "leveraged"]
    for inst in eligible.instId.dropna().astype(str).unique():
        attempted += 1
        try:
            update_cache(client, inst, "5m", HISTORY_DAYS, DATA)
            count += 1
        except (OKXError, requests.RequestException, ValueError) as exc:
            failures += 1
            print(f"candle {inst} refresh failed: {exc}", flush=True)
    if attempted and failures == attempted:
        raise RuntimeError(f"all {attempted} stock candle requests failed")
    return count


def _write_status(status: dict) -> None:
    """Atomically publish source heartbeats for the trading loop/monitoring."""
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATUS.with_suffix(STATUS.suffix + ".tmp")
    temporary.write_text(json.dumps(status, ensure_ascii=True, indent=2), encoding="utf-8")
    temporary.replace(STATUS)


def _latest_candle_ts() -> str | None:
    """Return the newest persisted confirmed bar without trusting file mtime."""
    paths = list((DATA / "5m").glob("*.csv")) if (DATA / "5m").exists() else []
    newest = None
    for path in paths:
        try:
            tail = pd.read_csv(path, usecols=["ts"]).tail(1)
            if tail.empty:
                continue
            stamp = pd.to_datetime(tail.iloc[0]["ts"], utc=True, errors="coerce")
            if pd.notna(stamp) and (newest is None or stamp > newest):
                newest = stamp
        except (OSError, ValueError, KeyError):
            continue
    return newest.isoformat() if newest is not None else None


def run_once(do_sec: bool = True, do_candles: bool = True, status: dict | None = None) -> dict:
    status = status or {}
    stamp = datetime.now(timezone.utc).isoformat()
    status.setdefault("service", "okx-stock-data")
    status["updated_at"] = stamp
    status["sec_interval_s"] = SEC_INTERVAL
    status["candle_interval_s"] = CANDLE_INTERVAL
    if do_sec:
        try:
            sec_count = refresh_sec()
            sec = status.setdefault("sec", {})
            sec.update({"last_success_at": stamp, "last_error": None, "new_rows": sec_count})
            if FILINGS.exists():
                frame = pd.read_csv(FILINGS, usecols=["accepted"])
                accepted = pd.to_datetime(frame["accepted"], utc=True, errors="coerce",
                                          format="mixed").dropna()
                sec["latest_accepted_at"] = accepted.max().isoformat() if not accepted.empty else None
        except Exception as exc:  # keep candle refresh alive if SEC is unavailable
            status.setdefault("sec", {}).update({"last_error": str(exc), "last_failed_at": stamp})
            print(f"SEC refresh failed: {exc}", flush=True)
        _write_status(status)
    if do_candles:
        try:
            candle_count = refresh_candles(OKXClient(timeout=20))
            status.setdefault("candles", {}).update({
                "last_success_at": stamp, "last_error": None,
                "refreshed": candle_count, "latest_bar_at": _latest_candle_ts(),
            })
        except Exception as exc:  # keep the daemon alive across transient outages
            status.setdefault("candles", {}).update({"last_error": str(exc), "last_failed_at": stamp})
            print(f"candle refresh failed: {exc}", flush=True)
        _write_status(status)
    print(json.dumps({"ts": stamp,
                      "sec_new": status.get("sec", {}).get("new_rows", 0) if do_sec else None,
                      "candles_refreshed": status.get("candles", {}).get("refreshed", 0) if do_candles else None,
                      "sec_due": do_sec, "candles_due": do_candles,
                      "sec_interval_s": SEC_INTERVAL, "candle_interval_s": CANDLE_INTERVAL},
                     ensure_ascii=True), flush=True)
    return status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="refresh once and exit")
    args = parser.parse_args()
    status: dict = {}
    next_sec = next_candles = 0.0
    while True:
        now = time.monotonic()
        do_sec, do_candles = now >= next_sec, now >= next_candles
        if do_sec:
            next_sec = now + SEC_INTERVAL
        if do_candles:
            next_candles = now + CANDLE_INTERVAL
        status = run_once(do_sec=do_sec, do_candles=do_candles, status=status)
        if args.once:
            return
        wait = min(next_sec, next_candles) - time.monotonic()
        time.sleep(max(1.0, wait))


if __name__ == "__main__":
    main()
