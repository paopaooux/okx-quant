"""Requery missing research bars; never fill from unconfirmed prices."""
import json
from pathlib import Path

import pandas as pd

from scripts.analysis.download_stock_history import pace
from strategies.stocks.market.okx import OKXClient

ROOT = Path("data/research_stock_history_20260928")


def aggregate_minutes(batch, stamp):
    by_ts = {int(r[0]): r for r in batch if str(r[8]) == "1"}
    wanted = [int(t.timestamp()*1000) for t in pd.date_range(stamp, periods=5, freq="min")]
    if not all(t in by_ts for t in wanted):
        return None
    rows = [by_ts[t] for t in wanted]
    return dict(ts=stamp, open=float(rows[0][1]), high=max(float(r[2]) for r in rows),
                low=min(float(r[3]) for r in rows), close=float(rows[-1][4]),
                volume=sum(float(r[5]) for r in rows), volume_quote=sum(float(r[7]) for r in rows))


def main():
    manifest = json.loads((ROOT / "download_manifest.json").read_text())
    client = OKXClient(timeout=20, proxy_url="http://127.0.0.1:7890")
    records = []
    try:
        for item in manifest["completed"]:
            if not item["missing_bars"]:
                continue
            path = ROOT / "5m" / f"{item['inst']}.csv"
            frame = pd.read_csv(path, parse_dates=["ts"])
            start = max(pd.Timestamp(manifest["start"]), pd.Timestamp(item["listed"]).floor("5min"))
            expected = pd.date_range(start, pd.Timestamp(manifest["end"])-pd.Timedelta(minutes=5), freq="5min")
            missing = expected.difference(pd.DatetimeIndex(frame.ts))
            if len(missing) > 100:
                raise ValueError(f"Too many gaps to point-repair: {item['inst']}")
            for stamp in missing:
                record = dict(inst=item["inst"], ts=str(stamp))
                repaired = None
                for bar in ("5m", "1m"):
                    pace()
                    batch = client._get("/api/v5/market/history-candles", dict(
                        instId=item["inst"], bar=bar, limit="10",
                        after=str(int((stamp+pd.Timedelta(minutes=5)).timestamp()*1000))))
                    record[bar] = batch
                    if bar == "5m":
                        matches = [r for r in batch if int(r[0]) == int(stamp.timestamp()*1000) and str(r[8]) == "1"]
                        if matches:
                            r = matches[0]
                            repaired = dict(ts=stamp, open=float(r[1]), high=float(r[2]), low=float(r[3]),
                                            close=float(r[4]), volume=float(r[5]), volume_quote=float(r[7]))
                    else:
                        repaired = aggregate_minutes(batch, stamp)
                    if repaired:
                        record["repair_source"] = bar
                        frame = pd.concat([frame, pd.DataFrame([repaired])], ignore_index=True)
                        break
                record["repaired"] = repaired is not None
                records.append(record)
                print(item["inst"], stamp, "repaired" if repaired else "unresolved; left absent", flush=True)
            frame.sort_values("ts").drop_duplicates("ts").to_csv(path, index=False)
    finally:
        client.session.close()
        (ROOT / "gap_rechecks.json").write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
