"""Small, conservative automatic trader for the configured OKX account.

The execution venue is OKX throughout: completed OKX 15m candles provide the
bar clock and signal prices, OKX ticker prices drive exits, and the OKX client
places the resulting orders. Some Binance-only positioning fields
are unavailable on OKX and remain missing values; LightGBM handles those fields
explicitly rather than fabricating cross-venue proxies.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import uuid
import fcntl
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal, InvalidOperation, ROUND_DOWN
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import requests

from scripts.data import build
from scripts.live.okx_demo import BASE, SIMULATED_TRADING, DemoClient, OKXAPIError, save_balance, save_strategy_snapshots
from scripts.live.execution_risk import (
    TERMINAL, cleanup_protection, definitely_absent, defer_retry, protect_positions, release_entry,
    recover_missing_exit,
)
from strategies.stocks.market import data as stock_data
from strategies.stocks.market import sessions as stock_sessions
from strategies.stocks.config import Config as StockConfig
from strategies.stocks.research import events as stock_events
from scripts.live.combination_policy import (
    POLICY, entry_metadata, entry_rejection, exit_reason, migrate_policy_state, stock_instruments,
)

ROOT = Path(__file__).resolve().parents[2]
DATA, RESULTS, MODELS = ROOT / "data", ROOT / "results" / "crypto", ROOT / "models"
# Frozen production universe. Keep this order independent from the LightGBM
# categorical codes; build_crypto_signals uses SYMBOL_CODE below.
SYMBOLS = (
    "ADAUSDT", "BNBUSDT", "BTCUSDT", "DOGEUSDT",
    "ETHUSDT", "LINKUSDT", "SOLUSDT", "XRPUSDT",
)
SYMBOL_CODE = {symbol: i for i, symbol in enumerate(SYMBOLS)}
STOCK_DATA = Path(os.environ.get("AUTO_STOCK_DATA", DATA / "stocks_swap"))
STOCK_STATUS = Path(os.environ.get("AUTO_STOCK_STATUS", STOCK_DATA / "data_status.json"))
STOCK_DATA_MAX_AGE = max(120, int(os.environ.get("STOCK_DATA_MAX_AGE", "900")))
BAR_MS = 15 * 60 * 1000
CONFIG = os.environ.get("AUTO_CONFIG", "c")
TAIL = float(os.environ.get("AUTO_TAIL", "0.01"))
MODEL_TAG = os.environ.get("AUTO_MODEL_TAG", f"{CONFIG}_roll730")
POSITION_MODE = os.environ.get("AUTO_POSITION_MODE", "net_mode").strip().lower()
SIZE = os.environ.get("AUTO_SIZE", "0.01")
MAX_HOLD_BARS = int(os.environ.get("AUTO_MAX_HOLD_BARS", "48"))
INTERVAL = max(15, int(os.environ.get("AUTO_INTERVAL", "30")))
EXIT_INTERVAL = max(1, int(os.environ.get("AUTO_EXIT_INTERVAL", "5")))
CAPITAL_BUFFER = max(.005, float(os.environ.get("AUTO_CAPITAL_BUFFER", "0.01")))
MAX_DATA_AGE = int(os.environ.get("AUTO_MAX_DATA_AGE", "1800"))
METRICS_MAX_AGE = MAX_DATA_AGE + BAR_MS // 1000  # build.py deliberately shifts metrics one bar.
TICKER_MAX_AGE = min(30, MAX_DATA_AGE)
STATE_PATH = Path(os.environ.get("AUTO_STATE", DATA / "auto_demo_state.json"))
# Shared five-slot pool across stock and crypto contracts. Both directions are
# eligible, but each instrument has one net position; this is a capacity limit,
# not a 50/50 long-short split.
MAX_POSITIONS = max(1, int(os.environ.get("AUTO_MAX_POSITIONS", "5")))
# The combination backtest reserves one fifth of equity per accepted trade in
# the shared five-slot pool. A signal may reverse an instrument only after its
# existing net position has been closed.
SLOT_WEIGHT = float(os.environ.get("AUTO_SLOT_WEIGHT", "0.20"))
DYNAMIC_SIZE = os.environ.get("AUTO_DYNAMIC_SIZE", "true").strip().lower() in {
    "1", "true", "yes", "on"
}
INSTRUMENT_CACHE_SECONDS = max(30, int(os.environ.get("AUTO_INSTRUMENT_CACHE_SECONDS", "300")))
STOCK_SIZE = os.environ.get("AUTO_STOCK_SIZE", "1")
STOCK_STOP_BPS = float(os.environ.get("AUTO_STOCK_STOP_BPS", "300"))
STOCK_MAX_HOLD_HOURS = float(os.environ.get("AUTO_STOCK_MAX_HOLD_HOURS", "30"))
_last_stock_feed_warning = 0.0
_instrument_cache: dict[str, dict] = {}
_instrument_cache_at = 0.0
_stock_signal_cache_key = None
_stock_signal_cache: dict[str, dict] = {}


def order_pos_side(side: str) -> str | None:
    """One-way OKX mode does not send a hedge ``posSide`` parameter."""
    return None


def position_key(inst_id: str, side: str) -> str:
    """Use instrument and direction for state identity, including net mode."""
    return f"{inst_id}|{side}"


def _migrate_state(state: dict) -> dict:
    """Migrate pre-hedge state that used a bare symbol as its key."""
    positions = state.get("positions")
    if not isinstance(positions, dict):
        state["positions"] = {}
        return state
    migrated = {}
    for old_key, row in positions.items():
        if not isinstance(row, dict):
            continue
        inst = str(row.get("inst_id") or "")
        if not inst:
            inst = old_key.replace("USDT", "-USDT-SWAP") if old_key in SYMBOLS else old_key
        side = str(row.get("side") or "").lower()
        if side not in {"long", "short"}:
            continue
        row.setdefault("symbol", old_key if old_key in SYMBOLS else inst)
        migrated[position_key(inst, side)] = row
    state["positions"] = migrated
    return state


class LiveData:
    def __init__(self, session: requests.Session):
        self.s = session

    def _okx(self, path: str, params: dict) -> dict:
        """GET a public OKX endpoint with a small 429 backoff."""
        for attempt in range(4):
            r = self.s.get(BASE + path, params=params, timeout=15)
            if r.status_code != 429 or attempt == 3:
                r.raise_for_status()
                return r.json()
            time.sleep(0.7 * (attempt + 1))
        raise RuntimeError("unreachable")

    def klines(self, sym: str) -> pd.DataFrame:
        """Use OKX candles for live price features and execution alignment."""
        inst = sym.replace("USDT", "-USDT-SWAP")
        rows = []
        # OKX caps each response at 300 bars. Walk backwards so all rolling
        # features (including the 384-bar z-scores) use OKX history rather
        # than silently inheriting the local Binance archive.
        after = None
        for _ in range(5):
            params = {"instId": inst, "bar": "15m", "limit": "300"}
            if after is not None:
                params["after"] = str(after)
            payload = self._okx("/api/v5/market/history-candles", params)
            if payload.get("code") != "0":
                raise RuntimeError(payload)
            page = payload.get("data", [])
            if not page:
                break
            rows.extend(page)
            oldest = min(int(row[0]) for row in page)
            if after == oldest:
                break
            after = oldest
        live = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close",
                                           "volume", "vol_ccy", "vol_quote", "confirm"])
        live = live[live["confirm"].astype(str) == "1"].copy()
        for col in ("ts", "open", "high", "low", "close", "volume", "vol_quote"):
            live[col] = pd.to_numeric(live[col], errors="coerce")
        # OKX does not publish trade count/taker split. Use neutral half-volume
        # proxies for the price/volume features; unavailable positioning fields
        # are left as NaN by _okx_metrics rather than copied from another venue.
        live["quote_volume"] = live["vol_quote"]
        live["count"] = live["volume"]
        live["taker_buy_volume"] = live["volume"] * 0.5
        live["taker_buy_quote_volume"] = live["quote_volume"] * 0.5
        live = live[["ts", "open", "high", "low", "close", "volume", "quote_volume",
                     "count", "taker_buy_volume", "taker_buy_quote_volume"]]
        return live.drop_duplicates("ts", keep="last").dropna(subset=["ts", "close"]).sort_values("ts").reset_index(drop=True)

    def ticker(self, sym: str, asset_type: str = "crypto") -> tuple[float, int]:
        """Return the latest OKX traded price and its exchange timestamp."""
        inst = sym.replace("USDT", "-USDT-SWAP") if asset_type == "crypto" else sym
        payload = self._okx("/api/v5/market/ticker", {"instId": inst})
        if payload.get("code") != "0" or not payload.get("data"):
            raise RuntimeError(payload)
        row = payload["data"][0]
        price = float(row.get("last") or row.get("askPx") or row.get("bidPx") or 0.0)
        ts = int(row.get("ts") or 0)
        if price <= 0 or ts <= 0:
            raise RuntimeError(f"invalid OKX ticker for {sym}: {row}")
        return price, ts

    def metrics(self, sym: str) -> pd.DataFrame:
        return self._okx_metrics(sym)

    def _okx_metrics(self, sym: str) -> pd.DataFrame:
        """Fetch the OKX-native derivatives fields that are actually exposed.

        OKX publishes aggregate open interest and a global long/short account
        ratio. It does not publish the Binance top-trader or taker-ratio fields;
        those columns remain NaN so a Binance-trained model cannot mistake a
        fabricated proxy for an OKX observation.
        """
        inst = sym.replace("USDT", "-USDT-SWAP")
        ccy = sym.replace("USDT", "")
        oi_pages = []
        for period in ("1H", "5m"):
            p = self._okx("/api/v5/rubik/stat/contracts/open-interest-history",
                          {"instId": inst, "period": period, "limit": "100"})
            if p.get("code") != "0":
                raise RuntimeError(p)
            oi_pages.extend(p.get("data", []))
        oi = pd.DataFrame(oi_pages, columns=["ts", "sum_open_interest",
                                              "oi_ccy", "sum_open_interest_value"])
        ratio_pages = []
        for period in ("1H", "5m"):
            p = self._okx("/api/v5/rubik/stat/contracts/long-short-account-ratio",
                          {"ccy": ccy, "instType": "SWAP", "period": period, "limit": "100"})
            if p.get("code") != "0":
                raise RuntimeError(p)
            ratio_pages.extend(p.get("data", []))
        ratio = pd.DataFrame(ratio_pages, columns=["ts", "count_long_short_ratio"])
        for frame in (oi, ratio):
            frame["ts"] = pd.to_numeric(frame["ts"], errors="coerce")
            if frame.empty or not fresh_bar(frame["ts"].max()):
                raise RuntimeError(f"stale or empty OKX metrics for {sym}")
        for col in ("sum_open_interest", "sum_open_interest_value", "count_long_short_ratio"):
            if col in oi:
                oi[col] = pd.to_numeric(oi[col], errors="coerce")
            if col in ratio:
                ratio[col] = pd.to_numeric(ratio[col], errors="coerce")
        oi["oi_source_ts"] = oi["ts"]
        ratio["ratio_source_ts"] = ratio["ts"]
        m = oi[["ts", "sum_open_interest", "sum_open_interest_value"]].merge(
            ratio[["ts", "count_long_short_ratio"]], on="ts", how="outer").dropna(subset=["ts"])
        m["ts"] = (m["ts"].astype("int64") // BAR_MS) * BAR_MS
        m = m.groupby("ts", as_index=False).last().sort_values("ts")
        for col in ("count_toptrader_long_short_ratio", "sum_toptrader_long_short_ratio",
                    "sum_taker_long_short_vol_ratio"):
            m[col] = np.nan
        # Hold the hourly observations across the 15m bars used by build.py.
        grid = pd.DataFrame({"ts": np.arange(int(m.ts.min()), int(m.ts.max()) + BAR_MS, BAR_MS)})
        m = pd.merge_asof(grid, m.sort_values("ts"), on="ts", direction="backward")
        for frame, source in ((oi, "oi_source_ts"), (ratio, "ratio_source_ts")):
            sources = frame[[source]].dropna().drop_duplicates().sort_values(source)
            sources["ts"] = (sources[source].astype("int64") // BAR_MS) * BAR_MS
            m = pd.merge_asof(m.sort_values("ts"), sources.sort_values("ts"), on="ts", direction="backward")
        return m


def load_state() -> dict:
    if not STATE_PATH.exists():
        return {"positions": {}, "last_bar": 0, "trades": 0}
    try:
        return _migrate_state(json.loads(STATE_PATH.read_text(encoding="utf-8")))
    except (OSError, ValueError) as exc:
        raise RuntimeError("Cannot read live state; refusing to forget positions or entry counters") from exc


def save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, ensure_ascii=True, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(STATE_PATH)
    directory = os.open(STATE_PATH.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _allowed_instruments() -> set[str]:
    """Return only instruments owned by this strategy, never manual positions."""
    allowed = {sym.replace("USDT", "-USDT-SWAP") for sym in SYMBOLS}
    universe = STOCK_DATA / "universe.csv"
    try:
        frame = pd.read_csv(universe, usecols=["instId"])
        allowed.update(
            str(inst) for inst in frame["instId"].dropna()
            if str(inst).endswith("-USDT-SWAP")
        )
    except (OSError, ValueError, KeyError):
        pass
    return allowed


def remote_positions(client: DemoClient) -> dict[str, dict]:
    out = {}
    # This strategy is contract-only. Do not import spot/margin positions or
    # unrelated manual swaps into the shared pool.
    allowed = _allowed_instruments()
    rows = client.positions("SWAP")
    for row in rows:
        inst = row.get("instId", "")
        if inst not in allowed:
            continue
        pos = float(row.get("pos") or 0)
        if abs(pos) < 1e-12:
            continue
        if inst.endswith("-USDT-SWAP"):
            base = inst.removesuffix("-USDT-SWAP")
            # Crypto signal keys are the compact BTCUSDT form. Stock signal
            # keys stay as their exact swap instId so they match stock events.
            sym = f"{base}USDT" if f"{base}USDT" in SYMBOLS else inst
        else:
            sym = inst
        pos_side = str(row.get("posSide") or "").lower()
        direction = pos_side if pos_side in {"long", "short"} else ("long" if pos > 0 else "short")
        out[position_key(inst, direction)] = {
            "symbol": sym, "inst_id": inst, "side": direction,
            "size": str(abs(pos)), "entry_px": float(row.get("avgPx") or 0),
            "asset_type": "crypto" if sym in SYMBOLS else "stock",
            "opened_bar": int(row.get("cTime") or 0),
            "upl": float(row.get("upl") or 0),
            "mark_px": float(row.get("markPx") or row.get("avgPx") or 0),
            "margin": float(row.get("margin") or 0),
        }
    return out


def contract_specs(client: DemoClient) -> dict[str, dict]:
    """Cache live SWAP metadata used to convert notional into contract size."""
    global _instrument_cache, _instrument_cache_at
    now = time.monotonic()
    if _instrument_cache and now - _instrument_cache_at < INSTRUMENT_CACHE_SECONDS:
        return _instrument_cache
    rows = client.instruments("SWAP")
    _instrument_cache = {
        str(row.get("instId")): row for row in rows
        if str(row.get("instId", "")).endswith("-USDT-SWAP")
        and str(row.get("state", "live")) == "live"
    }
    _instrument_cache_at = now
    return _instrument_cache


def _decimal(value: object, default: str = "0") -> Decimal:
    try:
        result = Decimal(str(value))
        return result if result.is_finite() else Decimal(default)
    except (InvalidOperation, TypeError, ValueError):
        return Decimal(default)


def _format_size(size: Decimal, lot: Decimal) -> str:
    text = format(size, "f")
    if lot.as_tuple().exponent >= 0:
        return str(int(size))
    return text.rstrip("0").rstrip(".") or "0"


def size_for_signal(client: DemoClient, sig: dict, mark: float,
                    total_eq: object, specs: dict[str, dict], *,
                    round_up: bool = False, budget=None) -> str | None:
    """Return an OKX-lot-aligned size for one shared-pool slot.

    A configured fixed size is an opt-in fallback for read-only tests. Live
    trading skips a symbol when dynamic sizing or account metadata is missing.
    """
    fallback = STOCK_SIZE if sig.get("asset_type") == "stock" else SIZE
    if not DYNAMIC_SIZE:
        return fallback
    spec = specs.get(str(sig.get("inst_id")))
    equity = _decimal(total_eq)
    price = _decimal(mark)
    if not spec or equity <= 0 or price <= 0 or SLOT_WEIGHT <= 0:
        return None
    contract_notional = contract_value(spec, price)
    if contract_notional <= 0:
        return None
    lot = _decimal(spec.get("lotSz"))
    minimum = _decimal(spec.get("minSz"))
    if lot <= 0 or minimum <= 0:
        return None
    notional = equity * _decimal(str(SLOT_WEIGHT))
    if budget is not None:
        notional = min(notional, _decimal(budget))
    target = notional / contract_notional
    size = (target / lot).to_integral_value(rounding=ROUND_DOWN) * lot
    if size < minimum:
        return None
    return _format_size(size, lot)


def contract_value(spec, mark):
    value = _decimal(spec.get("ctVal")) * _decimal(spec.get("ctMult") or "1")
    if str(spec.get("ctValCcy", "")).upper() not in {"USD", "USDT"}:
        value *= _decimal(mark)
    return value


def entry_budget(balance, state, specs, tickers):
    usdt = next((d for d in (balance[0].get("details", []) if balance else [])
                 if d.get("ccy") == "USDT"), {})
    available = [_decimal(usdt[k]) for k in ("availEq", "availBal") if usdt.get(k) not in (None, "")]
    equity = _decimal(usdt.get("cashBal"))
    if not available or equity <= 0:
        return Decimal(0), Decimal(0)
    used = Decimal(0)
    for p in list(state["positions"].values()) + list(state.get("pending_entries", {}).values()):
        spec = specs.get(p["inst_id"])
        if not spec:
            return equity, Decimal(0)
        mark = tickers.get(p.get("symbol"), (p.get("mark_px") or p["entry_px"], 0))[0]
        value = max(contract_value(spec, mark), contract_value(spec, p["entry_px"]))
        if value <= 0:
            return equity, Decimal(0)
        used += max(value * _decimal(p["size"]), _decimal(p.get("margin")))
    reserve = equity * _decimal(CAPITAL_BUFFER)
    return equity, max(Decimal(0), min(equity - used, min(available)) - reserve)


def fresh_bar(stamp, *, max_age=None):
    try:
        age = time.time() - float(stamp) / 1000
        return -5 <= age <= (MAX_DATA_AGE if max_age is None else max_age)
    except (TypeError, ValueError):
        return False


def use_fixed_size_fallback() -> bool:
    """Keep the size fallback explicit and easy to audit in logs/tests."""
    return os.environ.get("AUTO_DYNAMIC_SIZE_FALLBACK", "fixed").strip().lower() == "fixed"


def validate_position_mode(client: DemoClient) -> str:
    """Fail closed when configured execution mode differs from the account."""
    if POSITION_MODE != "net_mode":
        raise RuntimeError("this strategy requires AUTO_POSITION_MODE=net_mode (one-way contracts)")
    rows = client.account_config()
    actual = str(rows[0].get("posMode") or "") if rows else ""
    if actual and actual != POSITION_MODE:
        raise RuntimeError(
            f"OKX account posMode={actual!r}, but strategy requires {POSITION_MODE!r}; "
            "change the account mode before enabling AUTO_TRADE"
        )
    if not actual:
        raise RuntimeError("OKX account config did not return posMode")
    return actual


def build_crypto_signals(data: LiveData) -> dict[str, dict]:
    raw_k, raw_m, frames = {}, {}, {}
    for sym in SYMBOLS:
        try:
            k, m = data.klines(sym), data.metrics(sym)
            if k.empty or m.empty or not fresh_bar(k.ts.max()) or not fresh_bar(m.ts.max()):
                raise RuntimeError("stale per-symbol input")
            raw_k[sym], raw_m[sym] = k, m
        except (requests.RequestException, RuntimeError, ValueError, KeyError) as exc:
            print(f"CRYPTO DATA SKIP {sym}: {type(exc).__name__}", flush=True)
    if "BTCUSDT" not in raw_k:
        return {}  # Altcoin relative-return features depend on fresh BTC bars.
    btc_close = raw_k["BTCUSDT"].set_index("ts")["close"].astype(float)
    btc_ret = pd.Series(np.log(btc_close).diff(16).to_numpy(), index=btc_close.index)
    for sym in raw_k:
        frames[sym] = build.build_symbol(sym, btc_ret if sym != "BTCUSDT" else None,
                                         klines_df=raw_k[sym], metrics_df=raw_m[sym])

    model = lgb.Booster(model_file=str(MODELS / f"dir_{MODEL_TAG}_fold5.txt"))
    feats = model.feature_name()
    oos = pd.read_csv(RESULTS / f"oos_dir_{MODEL_TAG}.csv.gz")
    last_fold = oos[oos.fold == oos.fold.max()]
    hi = float(last_fold[f"hi_{TAIL}"].iloc[-1])
    lo = float(last_fold[f"lo_{TAIL}"].iloc[-1])
    signals = {}
    cutoff = int(time.time() * 1000) // BAR_MS * BAR_MS
    for sym in frames:
        f = frames[sym]
        # LightGBM can route missing values. Requiring every feature to be
        # non-null would discard all OKX bars because OKX has no top-trader and
        # taker-ratio history; those fields are intentionally left as NaN.
        rows = f[f.ts < cutoff].dropna(subset=["sigma", "ret_96", "dist_ema_96", "dist_ema_384"])
        if rows.empty:
            continue
        row = rows.iloc[-1]
        if not fresh_bar(row["ts"]):
            continue
        aligned = raw_m[sym][raw_m[sym].ts < row["ts"]]
        if aligned.empty or not fresh_bar(aligned.ts.max(), max_age=METRICS_MAX_AGE):
            continue
        source_ts = [aligned.ts.max()]
        source_ts.extend(aligned.iloc[-1][c] for c in ("oi_source_ts", "ratio_source_ts") if c in aligned)
        if not all(fresh_bar(t, max_age=METRICS_MAX_AGE) for t in source_ts):
            continue
        if not fresh_bar(raw_k["BTCUSDT"].ts.max()):
            continue
        x = row[[name for name in feats if name != "sym_code"]].to_frame().T.copy()
        x["sym_code"] = SYMBOL_CODE[sym]
        x = x[feats]
        x = x.apply(pd.to_numeric, errors="coerce").astype(float)
        p = float(model.predict(x, num_iteration=model.current_iteration())[0])
        side = "long" if p >= hi else ("short" if p <= lo else "flat")
        close = float(raw_k[sym].set_index("ts").loc[int(row["ts"]), "close"])
        signals[sym] = {"bar": int(row["ts"]), "side": side, "p_up": p,
                        "close": close, "width": float(row[f"width_{CONFIG}" ]),
                        "strategy": "okx_quant_c", "asset_type": "crypto",
                        "inst_id": sym.replace("USDT", "-USDT-SWAP"),
                        "missing_features": x.columns[x.isna().iloc[0]].tolist()}
        signals[sym]["metrics_ts"] = int(min(source_ts))
    return signals


def build_stock_signals(now: pd.Timestamp) -> dict[str, dict]:
    """Build stock-perpetual signals from recent OKX off-hours dislocations."""
    global _last_stock_feed_warning, _stock_signal_cache_key, _stock_signal_cache
    try:
        status = json.loads(STOCK_STATUS.read_text(encoding="utf-8"))
        now_epoch = now.timestamp()
        updated = pd.to_datetime(status.get("updated_at"), utc=True).timestamp()
        candles_ok = pd.to_datetime(status.get("candles", {}).get("last_success_at"), utc=True).timestamp()
        if min(now_epoch - updated, now_epoch - candles_ok) < -5 or \
                max(now_epoch - updated, now_epoch - candles_ok) > STOCK_DATA_MAX_AGE:
            raise ValueError("stock feed heartbeat is stale")
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        if time.time() - _last_stock_feed_warning > STOCK_DATA_MAX_AGE:
            print(f"stock feed unavailable/stale: {STOCK_STATUS}", flush=True)
            _last_stock_feed_warning = time.time()
        return {}
    cache_key = status.get("updated_at")
    if cache_key == _stock_signal_cache_key:
        return {sym: sig for sym, sig in _stock_signal_cache.items()
                if now - pd.Timestamp(sig["event_ts"]) <= pd.Timedelta(seconds=STOCK_DATA_MAX_AGE)}
    try:
        universe_path = STOCK_DATA / "universe.csv"
        if not universe_path.exists():
            return {}
        universe = pd.read_csv(universe_path)
        inst_ids = stock_instruments(universe)
        ticker_by_inst = universe.set_index("instId")["ticker"].astype(str).to_dict()
        frames = stock_data.to_bar_end(stock_data.load_panel(inst_ids, "5m", STOCK_DATA), "5m")
        frames = {inst: frame.loc[(frame.index <= now) & (frame.index >= now - pd.Timedelta(days=18))]
                  for inst, frame in frames.items()}
        frames = {inst: frame for inst, frame in frames.items() if len(frame) and
                  now - frame.index[-1] <= pd.Timedelta(seconds=STOCK_DATA_MAX_AGE)}
        # Calendar look-ahead is known in advance; prices are strictly clipped
        # to now. Include the next cash open across weekends and holidays.
        windows = stock_sessions.closed_windows(now - pd.Timedelta(days=8), now + pd.Timedelta(days=8))
        windows = windows.loc[(windows.close_ts < now) & (windows.open_ts > now)]
        cfg = StockConfig(dislocation_bps=POLICY.stock_trigger_bps, data_dir=str(STOCK_DATA),
                          result_dir=str(ROOT / "results" / "stocks_swap"))
        events = stock_events.off_hours_dislocation(frames, cfg, windows=windows)
        if events.empty:
            _stock_signal_cache_key, _stock_signal_cache = cache_key, {}
            return {}
        out = {}
        for row in events.sort_values("event_ts").itertuples(index=False):
            event_ts = pd.Timestamp(row.event_ts)
            # Never enter a position on a stale event after a restart. The
            # event must be newly actionable within one loop window; older
            # events remain useful in the backtest but are not live signals.
            if event_ts < now - pd.Timedelta(seconds=STOCK_DATA_MAX_AGE) or event_ts > now:
                continue
            inst = str(row.inst_id)
            ticker = ticker_by_inst.get(inst, inst.split("-", 1)[0])
            out[inst] = {
                "bar": int(event_ts.timestamp() * 1000),
                "side": "long" if int(row.side) > 0 else "short",
                "p_up": None, "close": float(frames[inst].loc[event_ts, "close"]),
                "width": POLICY.stock_stop_bps / 1e4,
                "strategy": "xstock_hybrid", "asset_type": "stock", "inst_id": inst,
                "event_ts": event_ts.isoformat(),
                "ticker": ticker, "resolve_ts": pd.Timestamp(row.resolve_ts).isoformat(),
                "observe_move_bps": float(abs(row.deviation) * 1e4),
            }
        _stock_signal_cache_key, _stock_signal_cache = cache_key, out
        return out
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"stock signal build failed: {exc}", flush=True)
        return {}


def signal_priority(sig: dict) -> float:
    """Rank simultaneous entries using information known at entry time."""
    if sig.get("asset_type") == "crypto" and sig.get("p_up") is not None:
        return abs(float(sig["p_up"]) - 0.5)
    return abs(float(sig.get("observe_move_bps") or 0.0)) / 10_000.0


def reconcile_pending_entries(client, state):
    state["entry_reconciliation_ok"] = True
    for client_id, pending in list(state.setdefault("pending_entries", {}).items()):
        try:
            detail = client.order_detail(pending["inst_id"], pending.get("ord_id"), client_order_id=client_id)
        except (requests.RequestException, RuntimeError, ValueError) as exc:
            # An uncertain submission reserves its slot. Never retry it under
            # a new ID, and do not let an entry lookup disable other exits.
            try:
                absent = definitely_absent(client, pending, client_id, exc)
            except (requests.RequestException, RuntimeError, ValueError):
                absent = False
            if absent:
                release_entry(state, client_id, retry=True, error=exc)
                save_state(state)
                continue
            state["entry_reconciliation_ok"] = False
            print(f"ENTRY RECONCILIATION UNKNOWN {pending['inst_id']} client_id={client_id}", flush=True)
            continue
        filled = float(detail.get("accFillSz") or detail.get("fillSz") or 0.)
        if filled > 0:
            key = position_key(pending["inst_id"], pending["side"])
            exiting = {k: v for k, v in state["positions"].get(key, {}).items()
                       if k.startswith(("exit_", "stop_")) or k == "force_exit"}
            state["positions"][key] = {**pending, "size": str(filled),
                                        "entry_px": float(detail.get("avgPx") or pending["entry_px"]), **exiting}
        if detail.get("state") in {"filled", "canceled", "mmp_canceled"}:
            if filled > 0:
                state["pending_entries"].pop(client_id)
                state["trades"] = int(state.get("trades", 0)) + 1
            else:
                release_entry(state, client_id, retry=True)
        elif detail.get("ordId"):
            # A partially filled market entry must not keep adding exposure
            # after its filled portion has been closed by risk management.
            try:
                client.cancel_order(pending["inst_id"], detail["ordId"])
            except (requests.RequestException, RuntimeError, ValueError):
                state["entry_reconciliation_ok"] = False
        save_state(state)


def manage_positions(client, state, data, now, allow_orders):
    """Exit management does not depend on a fresh entry signal or model run."""
    tickers = {}
    for key, pos in list(state["positions"].items()):
        sym, inst = pos["symbol"], pos["inst_id"]
        if pos.get("exit_ord_id") or pos.get("exit_client_id"):
            if not allow_orders:
                continue
            try:
                detail = client.order_detail(inst, pos.get("exit_ord_id"), client_order_id=pos.get("exit_client_id"))
            except (requests.RequestException, RuntimeError, ValueError) as exc:
                try:
                    recovered = recover_missing_exit(client, pos, exc)
                except (requests.RequestException, RuntimeError, ValueError):
                    recovered = False
                save_state(state)
                if recovered:
                    continue
                state["entry_reconciliation_ok"] = False
                print(f"EXIT RECONCILIATION UNKNOWN {sym}", flush=True)
                continue
            if detail.get("state") == "filled":
                state["positions"].pop(key, None)
                state["trades"] = int(state.get("trades", 0)) + 1
            elif detail.get("state") in {"canceled", "mmp_canceled"}:
                pos.pop("exit_ord_id", None)
                pos.pop("exit_client_id", None)
            save_state(state)
            continue
        reason = pos.get("force_exit") or exit_reason(pos, float("nan"), pd.Timestamp.now(tz="UTC"))
        if not reason:
            try:
                mark, stamp = data.ticker(sym, pos["asset_type"])
                if not fresh_bar(stamp, max_age=TICKER_MAX_AGE):
                    raise RuntimeError("stale execution ticker")
            except (requests.RequestException, RuntimeError, ValueError) as exc:
                print(f"EXIT PRICE UNAVAILABLE {sym}: {type(exc).__name__}", flush=True)
                continue
            tickers[sym] = (mark, stamp)
            reason = exit_reason(pos, mark, pd.Timestamp.now(tz="UTC"))
        if not reason or not allow_orders or time.time() < pos.get("exit_retry_after", 0):
            continue
        client_id = uuid.uuid4().hex
        pos["exit_client_id"] = client_id
        pos["exit_exp_time"] = int(time.time() * 1000) + 10_000
        save_state(state)
        try:
            result = client.order(inst, "sell" if pos["side"] == "long" else "buy", pos["size"],
                                  "isolated", True, pos_side=order_pos_side(pos["side"]), client_order_id=client_id,
                                  exp_time=pos["exit_exp_time"])
            pos["exit_ord_id"] = result[0].get("ordId") if result else None
            save_state(state)
            detail = client.order_detail(inst, pos.get("exit_ord_id"), client_order_id=client_id)
        except (requests.RequestException, RuntimeError, ValueError) as exc:
            if isinstance(exc, OKXAPIError) and exc.rejected and not pos.get("exit_ord_id"):
                pos.pop("exit_client_id", None)
                defer_retry(pos, "exit_retry", exc)
                save_state(state)
                print(f"EXIT REJECTED {sym}: {exc}", flush=True)
                continue
            state["entry_reconciliation_ok"] = False
            print(f"EXIT SUBMISSION UNKNOWN {sym} client_id={client_id}", flush=True)
            continue
        if detail.get("state") == "filled":
            state["positions"].pop(key, None)
            state["trades"] = int(state.get("trades", 0)) + 1
            print(f"EXIT {sym} reason={reason} policy={pos.get('policy_version')}", flush=True)
        elif detail.get("state") in {"canceled", "mmp_canceled"}:
            pos.pop("exit_ord_id", None)
            pos.pop("exit_client_id", None)
        save_state(state)
    return tickers


def run_once(client: DemoClient, state: dict, allow_orders: bool, *, prepared_signals=None,
             management_only=False) -> dict:
    state = _migrate_state(state)
    data = LiveData(client.s)
    now = pd.Timestamp.now(tz="UTC")
    # Reconcile before constructing tickers/signals so a position discovered
    # after a restart is still managed even when its original stock event has
    # left the current candle window.
    if allow_orders:
        reconcile_pending_entries(client, state)
    try:
        remote = remote_positions(client)
    except (requests.RequestException, RuntimeError, ValueError) as exc:
        state["entry_reconciliation_ok"] = False
        print(f"POSITION SYNC FAILED: {type(exc).__name__}; managing known positions", flush=True)
        migrate_policy_state(state, now)
        manage_positions(client, state, data, now, allow_orders)
        if allow_orders:
            if state["positions"]:
                try:
                    protect_positions(client, state, contract_specs(client), save_state)
                except (requests.RequestException, RuntimeError, ValueError) as protection_error:
                    print(f"PROTECTION UNAVAILABLE: {type(protection_error).__name__}", flush=True)
                if any(p.get("force_exit") for p in state["positions"].values()):
                    manage_positions(client, state, data, pd.Timestamp.now(tz="UTC"), True)
            save_state(state)
        return state
    pending_instruments = {p["inst_id"] for p in state.get("pending_entries", {}).values()}
    for key in list(state["positions"]):
        if key not in remote and state["positions"][key]["inst_id"] not in pending_instruments:
            state["positions"].pop(key, None)
    for key, pos in remote.items():
        if key in state["positions"]:
            state["positions"][key].update(size=pos["size"], upl=pos["upl"],
                                            mark_px=pos.get("mark_px"), margin=pos.get("margin"))
        else:
            pending = next((p for p in state.get("pending_entries", {}).values()
                            if p["inst_id"] == pos["inst_id"] and p["side"] == pos["side"]), {})
            state["positions"][key] = {**pos, "width": .06 if pos["asset_type"] == "stock" else 0.,
                                        **pending, "size": pos["size"], "entry_px": pos["entry_px"], "upl": pos["upl"]}
    migrate_policy_state(state, now)
    if allow_orders:
        save_state(state)
    managed_tickers = manage_positions(client, state, data, now, allow_orders)
    if allow_orders:
        cleanup_protection(client, state, save_state)
        if state["positions"]:
            try:
                protect_positions(client, state, contract_specs(client), save_state)
            except (requests.RequestException, RuntimeError, ValueError) as exc:
                state["entry_reconciliation_ok"] = False
                print(f"PROTECTION METADATA UNAVAILABLE: {type(exc).__name__}", flush=True)
            if any(p.get("force_exit") for p in state["positions"].values()):
                manage_positions(client, state, data, pd.Timestamp.now(tz="UTC"), allow_orders)
    if management_only:
        return state
    # Exits always run first. An unsuccessful leverage repair blocks entries,
    # while the next cycle can still close positions normally.
    if allow_orders:
        for inst in sorted({p["inst_id"] for p in state["positions"].values()}):
            try:
                client.ensure_unleveraged(inst)
            except (requests.RequestException, RuntimeError, ValueError) as exc:
                state["entry_reconciliation_ok"] = False
                print(f"LEVERAGE CHECK FAILED {inst}: {exc}", flush=True)
    signals = dict(prepared_signals) if prepared_signals is not None else collect_signals(data)
    # Continue managing a live position after its original stock event leaves
    # the current candle window.
    for key, pos in state.get("positions", {}).items():
        sym = pos.get("symbol") or pos.get("inst_id") or key
        if sym not in signals:
            signals[sym] = {
                "bar": int(now.timestamp() * 1000) // BAR_MS * BAR_MS,
                "side": pos.get("side", "flat"), "p_up": None, "close": None,
                "width": float(pos.get("width") or STOCK_STOP_BPS / 1e4),
                "strategy": pos.get("strategy", "xstock_hybrid"),
                "asset_type": pos.get("asset_type", "stock"),
                "inst_id": pos.get("inst_id", sym),
                "management_only": True,
            }
    # A closed-bar signal is deliberately separate from the live execution
    # price. Stops and timeouts must react to the OKX market now, not to the
    # close of the previous 15m bar (which can be minutes old).
    ticker_symbols = list(SYMBOLS) + [s for s, x in signals.items() if x.get("asset_type") == "stock"]
    ticker_symbols.extend(
        pos.get("symbol") or pos.get("inst_id")
        for pos in state["positions"].values()
        if pos.get("symbol") not in ticker_symbols and pos.get("inst_id") not in ticker_symbols
    )
    ticker_symbols = list(dict.fromkeys(s for s in ticker_symbols if s))
    tickers = {}
    for sym in ticker_symbols:
        try:
            tick = data.ticker(sym, signals.get(sym, {}).get("asset_type", "crypto"))
            if not fresh_bar(tick[1], max_age=TICKER_MAX_AGE):
                raise RuntimeError("stale ticker")
            tickers[sym] = tick
        except (requests.RequestException, RuntimeError, ValueError):
            print(f"ENTRY PRICE SKIP {sym}", flush=True)
    ticker_symbols = list(tickers)
    specs = contract_specs(client) if allow_orders else {}
    last_bars = state.setdefault("last_bars", {})
    balance = client.balance()
    save_balance(balance)
    total_eq = balance[0].get("totalEq") if balance else None
    # Only settled USDT funds finance isolated USDT swaps.
    sizing_eq, remaining_budget = entry_budget(balance, state, specs, tickers)
    cycle_id = uuid.uuid4().hex
    ordered_signals = sorted(
        signals.items(),
        key=lambda item: (item[1]["bar"], -signal_priority(item[1]),
                          item[1].get("strategy", ""), item[0]),
    )
    for sym, sig in ordered_signals:
        inst = sig.get("inst_id") or sym.replace("USDT", "-USDT-SWAP")
        bar_key = position_key(inst, sig["side"]) if sig["side"] in {"long", "short"} else sym
        max_age = STOCK_DATA_MAX_AGE if sig.get("asset_type") == "stock" else MAX_DATA_AGE
        if sym not in tickers or not fresh_bar(sig["bar"], max_age=max_age):
            continue
        if sig.get("metrics_ts") and not fresh_bar(sig["metrics_ts"], max_age=METRICS_MAX_AGE):
            continue
        if any(p["inst_id"] == inst for p in state.get("protective_orders", {}).values()
               if p.get("position_key") not in state["positions"]):
            continue
        retry = state.setdefault("entry_retries", {}).get(bar_key, {})
        if time.time() < retry.get("retry_after", 0):
            continue
        if sig.get("management_only") or sig["side"] == "flat" or sig["bar"] <= int(last_bars.get(bar_key, 0)):
            continue
        last_bars[bar_key] = sig["bar"]
        reason = entry_rejection(state, sig, pd.Timestamp.now(tz="UTC"))
        if reason:
            print(f"ENTRY SKIP {sym}: {reason}", flush=True)
            continue
        if not state.get("entry_reconciliation_ok", True):
            continue
        entry_key = position_key(inst, sig["side"])
        if entry_key in state["positions"]:
            continue
        # In one-way/net mode an opposite signal cannot open alongside the
        # current position. It is eligible only after the close above removed
        # the existing state entry.
        if any(pos.get("inst_id") == inst for pos in state["positions"].values()):
            continue
        active_count = len(state["positions"]) + len(state.get("pending_entries", {}))
        if allow_orders and active_count < MAX_POSITIONS:
            side = "buy" if sig["side"] == "long" else "sell"
            _, fresh_budget = entry_budget(client.balance(), state, specs, tickers)
            remaining_budget = min(remaining_budget, fresh_budget)
            size = size_for_signal(client, sig, tickers[sym][0], sizing_eq, specs,
                                   budget=remaining_budget)
            if not size:
                print(f"ENTRY SKIP {sym}: no valid contract size for equity={total_eq}", flush=True)
                continue
            # Both asset classes are -USDT-SWAP contracts; stock cash/margin
            # modes would turn this into a different execution strategy.
            td_mode = "isolated"
            quick_mgn_type = None
            client_id = uuid.uuid4().hex
            pending = {"symbol": sym, "inst_id": inst, "side": sig["side"], "size": size,
                       "entry_px": tickers[sym][0], "width": sig["width"],
                       "strategy": sig.get("strategy"), "asset_type": sig["asset_type"], **entry_metadata(sig)}
            if pd.Timestamp(pending["deadline_ts"]) <= pd.Timestamp.now(tz="UTC"):
                continue
            # Never rely on the exchange's per-instrument default leverage.
            # Check before reserving quota or persisting an order intent.
            try:
                if client.pending_orders(inst):
                    print(f"ENTRY SKIP {sym}: exchange orders still pending", flush=True)
                    continue
                client.ensure_unleveraged(inst)
            except (requests.RequestException, RuntimeError, ValueError) as exc:
                print(f"ENTRY SKIP {sym}: cannot verify 1x leverage: {exc}", flush=True)
                continue
            if not fresh_bar(sig["bar"], max_age=max_age) or pd.Timestamp(pending["deadline_ts"]) <= pd.Timestamp.now(tz="UTC"):
                continue
            # Account checks can take seconds. Requote and cap the final size
            # immediately before persisting the order intent.
            quote = data.ticker(sym, sig["asset_type"])
            if not fresh_bar(quote[1], max_age=TICKER_MAX_AGE):
                continue
            tickers[sym] = quote
            size = size_for_signal(client, sig, quote[0], sizing_eq, specs, budget=remaining_budget)
            if not size:
                continue
            pending.update(size=size, entry_px=quote[0])
            pending["exp_time"] = int(time.time() * 1000) + 10_000
            if sig["asset_type"] == "stock":
                day = str(pd.Timestamp.now(tz="UTC").date())
                ledger = state.setdefault("stock_entries_by_day", {})
                ledger[day] = ledger.get(day, 0) + 1
                pending["quota_day"] = day
            state["pending_entries"][client_id] = pending
            save_state(state)
            try:
                result = client.order(inst, side, size, td_mode, False, quick_mgn_type,
                                      pos_side=order_pos_side(sig["side"]), client_order_id=client_id,
                                      exp_time=pending["exp_time"])
            except OKXAPIError as exc:
                if exc.rejected:
                    release_entry(state, client_id, retry=True, error=exc)
                    save_state(state)
                    print(f"ENTRY REJECTED {sym}: {exc}", flush=True)
                    continue
                raise
            remaining_budget = max(Decimal(0), remaining_budget - _decimal(size) * contract_value(specs[inst], tickers[sym][0]))
            ord_id = result[0].get("ordId") if result else ""
            pending["ord_id"] = ord_id
            save_state(state)
            detail = client.order_detail(inst, ord_id, client_order_id=client_id)
            if detail.get("state") == "filled":
                state["positions"][entry_key] = {**pending,
                    "size": str(detail.get("accFillSz") or size),
                    "entry_px": float(detail.get("avgPx") or tickers[sym][0])}
                state["pending_entries"].pop(client_id, None)
                state["trades"] = int(state.get("trades", 0)) + 1
                state["entry_retries"].pop(bar_key, None)
                protect_positions(client, state, specs, save_state)
                if state["positions"][entry_key].get("force_exit"):
                    manage_positions(client, state, data, pd.Timestamp.now(tz="UTC"), allow_orders)
                p_text = f"{sig['p_up']:.5f}" if sig.get("p_up") is not None else "n/a"
                print(f"ENTRY {sym} strategy={sig.get('strategy')} side={sig['side']} p={p_text} ord={ord_id}", flush=True)
            elif detail.get("state") in {"canceled", "mmp_canceled"} and not float(detail.get("accFillSz") or 0):
                release_entry(state, client_id, retry=True)
            save_state(state)
    for sym, sig in signals.items():
        if sig.get("management_only"):
            continue
        inst = sig.get("inst_id") or sym.replace("USDT", "-USDT-SWAP")
        key = position_key(inst, sig["side"]) if sig.get("side") in {"long", "short"} else sym
        if key in state.get("entry_retries", {}):
            continue
        last_bars[key] = max(int(last_bars.get(key, 0)), int(sig.get("bar") or 0))
    state["last_bar"] = max(last_bars.values(), default=int(state.get("last_bar", 0)))
    # Store a row for every symbol on every cycle, including flat symbols. This
    # makes exposure, signal drift, latency and mark-to-market history queryable
    # at the same cadence as the strategy loop.
    snapshot_rows = []
    for sym in ticker_symbols:
        sig = signals.get(sym, {})
        mark, ticker_ts = tickers[sym]
        positions = [pos for pos in state["positions"].values()
                     if pos.get("symbol") == sym or pos.get("inst_id") == sig.get("inst_id")]
        # Keep one flat row when there is no position, and one row for the
        # instrument's single net position when it is open.
        for pos in positions or [{}]:
            entry = float(pos.get("entry_px") or 0.0)
            width = float(pos.get("width") or sig.get("width") or 0.0)
            side = pos.get("side")
            snapshot_rows.append({
                "ts": pd.Timestamp.now(tz="UTC").isoformat(), "cycle_id": cycle_id,
                "symbol": sym, "strategy": sig.get("strategy"), "asset_type": sig.get("asset_type"),
                "bar_ts": int(sig.get("bar") or 0),
                "signal_side": sig.get("side", "unknown"), "p_up": sig.get("p_up"),
                "signal_close": sig.get("close"), "ticker_px": mark, "ticker_ts": int(ticker_ts),
                "position_side": side, "position_size": pos.get("size"), "entry_px": entry or None,
                "width": width or None,
                "stop_px": (entry * (1 - width) if side == "long" else entry * (1 + width)) if side and entry and width else None,
                "take_px": ((entry * (1 + float(pos.get("take_width", width)))) if side == "long" else
                            (entry * (1 - float(pos.get("take_width", width)))))
                            if side and entry and float(pos.get("take_width", width)) else None,
                "total_eq": total_eq,
                "raw_json": json.dumps({"signal": sig, "position": pos}, ensure_ascii=True),
            })
    save_strategy_snapshots(snapshot_rows)
    if allow_orders:
        save_state(state)
    print("signals:", json.dumps(signals, ensure_ascii=True),
          "okx_tickers:", json.dumps({s: {"last": p, "ts": ts} for s, (p, ts) in tickers.items()},
                                      ensure_ascii=True),
          "positions:", len(state["positions"]), flush=True)
    return state


def collect_signals(data):
    signals = {}
    try:
        signals.update(build_crypto_signals(data))
    except (requests.RequestException, RuntimeError, ValueError, KeyError) as exc:
        print(f"CRYPTO ENTRIES UNAVAILABLE: {type(exc).__name__}", flush=True)
    signals.update(build_stock_signals(pd.Timestamp.now(tz="UTC")))
    return signals


class SignalWorker:
    """Signal computation owns its session and never reads or writes live state."""
    def __init__(self, data):
        self.data = data
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="signals")
        self.future = None
        self.next_run = 0.

    def poll(self):
        result = None
        if self.future is not None and self.future.done():
            try:
                result = self.future.result()
            finally:
                self.future = None
                self.next_run = time.monotonic() + INTERVAL
        if self.future is None and time.monotonic() >= self.next_run:
            self.future = self.pool.submit(collect_signals, self.data)
        return result

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--observe", action="store_true", help="never send orders")
    args = ap.parse_args()
    validate_combination_config()
    client = DemoClient()
    position_mode = validate_position_mode(client)
    state = load_state()
    allow = os.environ.get("AUTO_TRADE", "false").lower() in {"1", "true", "yes", "on"} and not args.observe
    lock = None
    if allow:
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        lock = STATE_PATH.with_suffix(".lock").open("a")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("another execution process owns this live state")
        state = load_state()
    print(f"auto demo model={MODEL_TAG} universe={len(SYMBOLS)} tail={TAIL} "
          f"allow_orders={allow} interval={INTERVAL}s "
          f"shared_pool_slots={MAX_POSITIONS} slot_weight={SLOT_WEIGHT:.4f} "
          f"dynamic_size={DYNAMIC_SIZE} position_mode={position_mode} "
          f"leverage=1 margin_mode=isolated "
          f"max_data_age={MAX_DATA_AGE}s "
          f"price_source=OKX metrics_source=OKX simulated_header={int(SIMULATED_TRADING)}", flush=True)
    print("COMBINATION_POLICY " + json.dumps(POLICY.manifest(), sort_keys=True), flush=True)
    if args.once:
        run_once(client, state, allow)
        return
    signal_client = DemoClient()
    worker = SignalWorker(LiveData(signal_client.s))
    try:
        while True:
            started = time.monotonic()
            try:
                signals = worker.poll()
                run_once(client, state, allow, prepared_signals=signals or {},
                         management_only=signals is None)
            except (requests.RequestException, RuntimeError, ValueError, KeyError) as exc:
                print(f"auto cycle failed: {exc}", flush=True)
            time.sleep(max(.1, EXIT_INTERVAL - (time.monotonic() - started)))
    finally:
        worker.close()
        signal_client.s.close()
        if lock:
            lock.close()


def validate_combination_config():
    actual = (CONFIG, TAIL, MODEL_TAG, MAX_POSITIONS, SLOT_WEIGHT, DYNAMIC_SIZE,
              STOCK_STOP_BPS, STOCK_MAX_HOLD_HOURS, MAX_HOLD_BARS, POSITION_MODE)
    expected = (POLICY.crypto_config, POLICY.crypto_tail, "c_roll730", POLICY.shared_slots,
                POLICY.slot_weight, True, POLICY.stock_stop_bps, POLICY.stock_max_hours,
                POLICY.crypto_horizon_bars, "net_mode")
    if actual != expected:
        raise RuntimeError(f"Execution configuration differs from frozen combination: {actual!r} != {expected!r}")


if __name__ == "__main__":
    main()
