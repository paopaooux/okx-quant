"""Bulk-pull Binance USDT-M perpetual 15m klines + 5m derivatives metrics.

Why the bulk archive and not the REST API: the live `/futures/data/*` endpoints
(openInterestHist, topLongShortAccountRatio, ...) only serve the trailing 30
days.  data.binance.vision carries the full history.  Klines come as monthly
zips; metrics are daily-only (monthly metrics zips do not exist -- verified by
404 on BTCUSDT-metrics-2024-01.zip).

Venue matching matters: these metrics describe the Binance USDT-M perpetual, so
the price series must be the *same* contract on the *same* venue.  Pairing OKX
spot candles with Binance perp open interest would silently mix two books.

Output
  data/klines/{SYMBOL}.csv.gz    15m OHLCV + taker-buy split, ts = bar open (UTC ms)
  data/metrics/{SYMBOL}.csv.gz   5m metrics folded to 15m, ts = bar open (UTC ms)
"""
from __future__ import annotations

import io
import sys
import time
import zipfile
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

BASE = "https://data.binance.vision/data/futures/um"
UA = {"User-Agent": "Mozilla/5.0"}
DEFAULT_SYMBOLS = (
    "ADAUSDT", "BNBUSDT", "BTCUSDT", "DOGEUSDT",
    "ETHUSDT", "LINKUSDT", "SOLUSDT", "XRPUSDT",
)
# Keep the original three-symbol default, while allowing a research run to
# widen the universe without editing source.  Example: --symbols=BNBUSDT,XRPUSDT.
_symbols_arg = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--symbols=")), None)
SYMBOLS = tuple(s.strip().upper() for s in _symbols_arg.split(",") if s.strip()) if _symbols_arg else DEFAULT_SYMBOLS
# 2021-01 is the earliest month with daily `metrics` zips for all three symbols
# (checked by HEAD: 2021-01-01 through 2023-06-01 all return 200).  The original
# 2023-09 start was chosen for convenience and it cost the study its regime
# coverage: the resulting out-of-sample window (2024-11 -> 2026-08) contains no
# sustained bull market, so a short-only signal could never be falsified in one.
# Starting in 2021 puts the 2021 top, the 2022 bear and the 2024 bull inside the
# sample.  Override with --start=YYYY-MM-DD.
import sys as _sys
_arg = next((a.split("=")[1] for a in _sys.argv if a.startswith("--start=")), None)
START = date.fromisoformat(_arg) if _arg else date(2021, 1, 1)
ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "cache"
OUT = ROOT / "data"
WORKERS = 8

KLINE_COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
              "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore"]


def fetch(url: str, retries: int = 4) -> bytes | None:
    """Download with a disk cache.

    A daily Binance archive can return 404 for several hours before it is
    published. Do not permanently negative-cache that response: otherwise a
    later refresh can never repair the newest one or two days of the archive.
    """
    key = CACHE / url.rsplit("/", 1)[-1]
    if key.exists():
        if key.stat().st_size:
            return key.read_bytes()
        # Empty files are legacy negative-cache entries. Retry them after a
        # short cooldown so historical missing files do not cause request
        # storms while newly published daily files are eventually picked up.
        if time.time() - key.stat().st_mtime < 6 * 3600:
            return None
        try:
            key.unlink()
        except OSError:
            return None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=60) as r:
                blob = r.read()
            key.write_bytes(blob)
            return blob
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(1.5 * (attempt + 1))
        except Exception:
            time.sleep(1.5 * (attempt + 1))
    print(f"  ! giving up on {url}", file=sys.stderr)
    return None


def unzip_csv(blob: bytes) -> pd.DataFrame | None:
    """Binance switched to writing a header row partway through 2025; handle both."""
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        raw = z.read(z.namelist()[0])
    if not raw.strip():
        return None
    head = raw[:16].decode("utf8", "ignore")
    has_header = head.startswith("open_time") or head.startswith("create_time")
    return pd.read_csv(io.BytesIO(raw), header=0 if has_header else None)


def months(start: date, end: date) -> list[str]:
    out, y, m = [], start.year, start.month
    while (y, m) <= (end.year, end.month):
        out.append(f"{y:04d}-{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def klines(symbol: str, today: date) -> pd.DataFrame:
    """Monthly zips plus daily zips for the newest (possibly unpublished) month.

    Binance can lag in publishing the just-completed monthly archive.  Keep the
    newest month on daily files so a run on the first day of a month still
    includes all bars through yesterday.
    """
    last_month_start = (today.replace(day=1) - timedelta(days=1)).replace(day=1)
    monthly_end = last_month_start - timedelta(days=1)
    urls = [f"{BASE}/monthly/klines/{symbol}/15m/{symbol}-15m-{ym}.zip"
            for ym in months(START, monthly_end)]
    d = max(START, last_month_start)
    while d < today:                       # today's file is written after UTC midnight
        urls.append(f"{BASE}/daily/klines/{symbol}/15m/{symbol}-15m-{d:%Y-%m-%d}.zip")
        d += timedelta(days=1)

    with ThreadPoolExecutor(WORKERS) as ex:
        blobs = list(ex.map(fetch, urls))

    frames = []
    for blob in blobs:
        if blob is None:
            continue
        df = unzip_csv(blob)
        if df is None:
            continue
        df.columns = KLINE_COLS[:df.shape[1]]
        frames.append(df)
    if not frames:
        raise RuntimeError(f"no kline data for {symbol}")

    k = pd.concat(frames, ignore_index=True)
    k = k[["open_time", "open", "high", "low", "close", "volume",
           "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume"]]
    k = k.rename(columns={"open_time": "ts"})
    k["ts"] = pd.to_numeric(k["ts"], errors="coerce").astype("Int64")
    # A few 2025 monthly files carry microsecond timestamps instead of ms.
    k.loc[k["ts"] > 10**14, "ts"] //= 1000
    for c in k.columns:
        if c != "ts":
            k[c] = pd.to_numeric(k[c], errors="coerce")
    return k.dropna(subset=["ts", "close"]).drop_duplicates("ts").sort_values("ts").reset_index(drop=True)


def metrics(symbol: str, today: date) -> pd.DataFrame:
    urls, d = [], START
    while d < today:
        urls.append(f"{BASE}/daily/metrics/{symbol}/{symbol}-metrics-{d:%Y-%m-%d}.zip")
        d += timedelta(days=1)

    with ThreadPoolExecutor(WORKERS) as ex:
        blobs = list(ex.map(fetch, urls))

    cols = ["create_time", "symbol", "sum_open_interest", "sum_open_interest_value",
            "count_toptrader_long_short_ratio", "sum_toptrader_long_short_ratio",
            "count_long_short_ratio", "sum_taker_long_short_vol_ratio"]
    frames = []
    for blob in blobs:
        if blob is None:
            continue
        df = unzip_csv(blob)
        if df is None or df.shape[1] < len(cols):
            continue
        df = df.iloc[:, :len(cols)]
        df.columns = cols
        frames.append(df)
    if not frames:
        raise RuntimeError(f"no metrics for {symbol}")

    m = pd.concat(frames, ignore_index=True).drop(columns=["symbol"])
    m["ts"] = pd.to_datetime(m["create_time"], utc=True, errors="coerce")
    m = m.drop(columns=["create_time"]).dropna(subset=["ts"])
    for c in m.columns:
        if c != "ts":
            m[c] = pd.to_numeric(m[c], errors="coerce")
    m = m.drop_duplicates("ts").sort_values("ts").set_index("ts")

    # Fold 5m -> 15m.  Open interest and the two positioning ratios are *levels*
    # (a snapshot of the book), so the bar's value is its last observation.  The
    # taker buy/sell ratio is a *flow* measured over each 5m window, so the bar's
    # value is the mean of its three windows.  n5m records how many of the three
    # 5m rows actually existed -- Binance drops rows during outages.
    g = m.resample("15min", label="left", closed="left")
    lvl = ["sum_open_interest", "sum_open_interest_value",
           "count_toptrader_long_short_ratio", "sum_toptrader_long_short_ratio",
           "count_long_short_ratio"]
    out = g[lvl].last()
    out["sum_taker_long_short_vol_ratio"] = g["sum_taker_long_short_vol_ratio"].mean()
    out["n5m"] = g["sum_open_interest"].count()
    out = out[out["n5m"] > 0].reset_index()
    # Pandas 3 may store timezone-aware datetimes at second or microsecond
    # resolution depending on the input.  Converting the raw integer and
    # always dividing by 1e6 therefore turns valid 2021 timestamps into 1970.
    # Normalize the datetime explicitly before serializing to Binance's
    # millisecond timestamp convention used by the kline files.
    out["ts"] = out["ts"].dt.as_unit("ms").astype("int64")
    return out


def main() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    (OUT / "klines").mkdir(parents=True, exist_ok=True)
    (OUT / "metrics").mkdir(parents=True, exist_ok=True)
    today = datetime.now(timezone.utc).date()

    for sym in SYMBOLS:
        for kind, fn in (("klines", klines), ("metrics", metrics)):
            path = OUT / kind / f"{sym}.csv.gz"
            # --force rebuilds even when the output exists.  Needed whenever START
            # moves: the raw zips are cached per file, so a rebuild re-reads the
            # cache and only fetches the genuinely new months.
            if path.exists() and "--force" not in _sys.argv:
                print(f"{sym} {kind}: exists, skip  (--force to rebuild)")
                continue
            t0 = time.time()
            df = fn(sym, today)
            df.to_csv(path, index=False, compression="gzip")
            span = (pd.to_datetime(df['ts'], unit='ms', utc=True).min(),
                    pd.to_datetime(df['ts'], unit='ms', utc=True).max())
            print(f"{sym} {kind}: {len(df):,} rows  {span[0]:%Y-%m-%d} -> {span[1]:%Y-%m-%d}"
                  f"  [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
