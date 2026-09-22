"""Read-only account backfill with atomic page checkpoints and overlapping polls."""
from __future__ import annotations

import argparse
import json
import os
import time

from scripts.live.okx_demo import DemoClient, db_connect, save_account_data, save_balance


def sync_stream(client, stream, *, max_pages=20):
    with db_connect() as db:
        row = db.execute("SELECT raw_json FROM account_sync_state WHERE stream=?", (stream,)).fetchone()
    checkpoint = json.loads(row[0]) if row else {}
    now = int(time.time() * 1000)
    if not checkpoint.get("active"):
        watermark = int(checkpoint.get("watermark", 0))
        checkpoint.update(active=True, begin=max(0, watermark - 3600_000) if watermark else
                          now - 89 * 86400_000, end=now, after=None)
    count = 0
    for _ in range(max_pages):
        params = {"begin": str(checkpoint["begin"]), "end": str(checkpoint["end"])}
        if checkpoint.get("after"):
            params["after"] = checkpoint["after"]
        rows = client.history_page(stream, **params)
        if rows:
            cursor = str(min(int(r["billId"]) for r in rows))
            if checkpoint.get("after") and int(cursor) >= int(checkpoint["after"]):
                raise RuntimeError(f"{stream} history cursor did not advance")
            checkpoint["after"] = cursor
        # Only an empty page completes a window, including after short pages.
        if not rows:
            checkpoint.update(active=False, watermark=checkpoint["end"], completed_at=now, after=None)
        save_account_data([], rows if stream == "fills" else [], rows if stream == "bills" else [],
                          sync_state=(stream, checkpoint))
        count += len(rows)
        if not rows:
            break
        time.sleep(.25)
    return count, not checkpoint["active"]


def sync_once(client):
    succeeded = True
    for stream in ("fills", "bills"):
        try:
            count, complete = sync_stream(client, stream)
            print(f"account sync {stream}: rows={count} window_complete={complete}", flush=True)
        except Exception as exc:
            succeeded = False
            print(f"account sync {stream} failed: {type(exc).__name__}: {exc}", flush=True)
    try:
        save_balance(client.balance())
        save_account_data(client.positions(), [], [])
    except Exception as exc:
        succeeded = False
        print(f"account snapshot failed: {type(exc).__name__}: {exc}", flush=True)
    return succeeded


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    client = DemoClient()
    interval = max(15, int(os.environ.get("OKX_SYNC_INTERVAL", "60")))
    while True:
        succeeded = sync_once(client)
        if args.once:
            raise SystemExit(0 if succeeded else 1)
        time.sleep(interval)


if __name__ == "__main__":
    main()
