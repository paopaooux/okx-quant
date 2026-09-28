"""Download confirmed public candles into an isolated, resumable archive."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time
import threading

import pandas as pd

from strategies.stocks.market.okx import OKXClient

_RATE_LOCK = threading.Lock()
_NEXT_REQUEST = 0.0


def pace():
    global _NEXT_REQUEST
    with _RATE_LOCK:
        now = time.monotonic()
        delay = max(0.0, _NEXT_REQUEST-now)
        _NEXT_REQUEST = max(now, _NEXT_REQUEST) + 1/6
    if delay:
        time.sleep(delay)


def download(inst, listed, root, start, end, proxy):
    path = root / "5m" / f"{inst}.csv"
    floor = max(start, listed.floor("5min"))
    old = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if not old.empty:
        old["ts"] = pd.to_datetime(old.ts, utc=True)
    cursor = int((old.ts.min() if not old.empty else end).timestamp() * 1000)
    pieces = [old] if not old.empty else []
    client = OKXClient(timeout=30, proxy_url=proxy)
    pages = 0

    def save():
        frame = pd.concat(pieces, ignore_index=True).drop_duplicates("ts", keep="last").sort_values("ts")
        frame = frame[frame.ts.ge(start) & frame.ts.lt(end)]
        temporary = path.with_suffix(".tmp")
        frame.to_csv(temporary, index=False)
        temporary.replace(path)
        pieces[:] = [frame]
        return frame

    try:
        while cursor > int(floor.timestamp() * 1000):
            pace()
            batch = client._get("/api/v5/market/history-candles", {
                "instId": inst, "bar": "5m", "after": str(cursor), "limit": "300"})
            if not batch:
                break
            oldest = min(int(r[0]) for r in batch)
            if oldest >= cursor:
                raise ValueError(f"Non-progressing pagination: {inst}")
            frame = pd.DataFrame([r for r in batch if str(r[8]) == "1"], columns=[
                "ts", "open", "high", "low", "close", "volume", "volume_ccy", "volume_quote", "confirm"])
            frame["ts"] = pd.to_datetime(frame.ts.astype("int64"), unit="ms", utc=True)
            frame = frame.drop(columns=["confirm", "volume_ccy"])
            for col in frame.columns.drop("ts"):
                frame[col] = pd.to_numeric(frame[col], errors="raise")
            pieces.append(frame)
            cursor = oldest
            pages += 1
            if pages % 25 == 0:
                save()
            time.sleep(.15)
        if not pieces:
            raise ValueError(f"No candles: {inst}")
        frame = save()
        expected = pd.date_range(floor, end-pd.Timedelta(minutes=5), freq="5min")
        missing = expected.difference(pd.DatetimeIndex(frame.ts))
        return dict(inst=inst, listed=str(listed), first=str(frame.ts.min()), last=str(frame.ts.max()),
                    rows=len(frame), missing_bars=len(missing), missing_first=[str(t) for t in missing[:5]],
                    pages=pages, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    finally:
        if pieces:
            save()
        client.session.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/research_stock_history_20260928"))
    parser.add_argument("--proxy", default="http://127.0.0.1:7890")
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    args.out.joinpath("5m").mkdir(parents=True, exist_ok=True)
    universe = pd.read_csv("data/stocks_swap/universe.csv")
    universe.to_csv(args.out / "universe.csv", index=False)
    start, end = pd.Timestamp("2026-02-25T00:00Z"), pd.Timestamp("2026-09-05T00:00Z")
    rows, errors = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download, r.instId, pd.Timestamp(r.list_ts), args.out,
                               start, end, args.proxy): r.instId for r in universe.itertuples()}
        for future in as_completed(futures):
            inst = futures[future]
            try:
                row = future.result()
                rows.append(row)
                print(f"{len(rows)}/{len(universe)} {inst}: {row['rows']} bars, {row['missing_bars']} missing", flush=True)
            except Exception as exc:
                errors.append(dict(inst=inst, error=str(exc)))
                print(f"FAILED {inst}: {exc}", flush=True)
            (args.out / "download_manifest.json").write_text(json.dumps(dict(
                start=str(start), end=str(end), completed=rows, errors=errors), indent=2))
    if errors:
        raise RuntimeError(f"{len(errors)} downloads failed; resumable files retained")


if __name__ == "__main__":
    main()
