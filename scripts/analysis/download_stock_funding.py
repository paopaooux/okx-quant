"""Freeze public realized funding histories for the reviewed stock universe."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import threading
import time

import pandas as pd

from strategies.stocks.market.okx import OKXClient
from strategies.stocks.market.reference import reference_market

ROOT = Path("data/research_stock_funding_20260928")
START = pd.Timestamp("2026-06-27T00:00Z")
END = pd.Timestamp("2026-09-27T00:00Z")
LOCK = threading.Lock()
NEXT = 0.


def pace():
    global NEXT
    with LOCK:
        now = time.monotonic()
        delay = max(0., NEXT-now)
        NEXT = max(now, NEXT)+.25
    if delay:
        time.sleep(delay)


def fetch(inst):
    path = ROOT / f"{inst}.json"
    client = OKXClient(timeout=25, proxy_url="http://127.0.0.1:7890")
    records = json.loads(path.read_text()) if path.exists() else []
    cursor = min((int(r["fundingTime"]) for r in records), default=int(END.timestamp()*1000))
    try:
        for _ in range(100):
            if cursor <= int(START.timestamp()*1000):
                break
            pace()
            batch = client._get("/api/v5/public/funding-rate-history", {
                "instId": inst, "limit": "100", "after": str(cursor)})
            if not batch:
                break
            oldest = min(int(r["fundingTime"]) for r in batch)
            if oldest >= cursor:
                raise ValueError(f"Non-progressing funding pagination: {inst}")
            records.extend(batch)
            cursor = oldest
        else:
            raise ValueError(f"Funding page limit reached: {inst}")
        records = sorted({r["fundingTime"]: r for r in records}.values(), key=lambda r: int(r["fundingTime"]))
        if not records or any(r.get("realizedRate", "") == "" for r in records):
            raise ValueError(f"Empty/missing realized rates: {inst}")
        path.write_text(json.dumps(records, indent=2))
        stamps = pd.to_datetime([int(r["fundingTime"]) for r in records], unit="ms", utc=True)
        return dict(inst=inst, records=len(records), first=str(stamps.min()), last=str(stamps.max()),
                    max_gap_h=float(stamps.to_series().diff().dt.total_seconds().max()/3600),
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    finally:
        client.session.close()


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    universe = pd.read_csv("data/stocks_swap/universe.csv")
    ids = sorted(inst for inst in universe.instId if reference_market(inst))
    completed, errors = [], []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(fetch, inst): inst for inst in ids}
        for future in as_completed(futures):
            try:
                result = future.result()
                completed.append(result)
                if len(completed) % 10 == 0 or len(completed) == len(ids):
                    print(f"Funding {len(completed)}/{len(ids)} complete", flush=True)
            except Exception as exc:
                errors.append(dict(inst=futures[future], error=str(exc)))
                print("Funding failed:", errors[-1], flush=True)
            (ROOT / "manifest.json").write_text(json.dumps(dict(start=str(START), end=str(END),
                completed=completed, errors=errors, source="OKX public funding-rate-history realizedRate"), indent=2))
    if errors:
        raise RuntimeError(f"{len(errors)} funding downloads failed")


if __name__ == "__main__":
    main()
