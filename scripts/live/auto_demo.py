"""Small, conservative automatic trader for the OKX simulated account.

The execution venue is OKX throughout: completed OKX 15m candles provide the
bar clock and signal prices, OKX ticker prices drive exits, and ``DemoClient``
places the resulting simulated orders. Some Binance-only positioning fields
are unavailable on OKX and remain missing values; LightGBM handles those fields
explicitly rather than fabricating cross-venue proxies.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import requests

from scripts.data import build
from scripts.live.okx_demo import BASE, SIMULATED_TRADING, DemoClient, save_balance, save_strategy_snapshots
from strategies.stocks.market import data as stock_data
from strategies.stocks.market.universe_tech import TECH
from strategies.stocks.research import news_strategy

ROOT = Path(__file__).resolve().parents[2]
DATA, RESULTS, MODELS = ROOT / "data", ROOT / "results" / "crypto", ROOT / "models"
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
STOCK_FILINGS = Path(os.environ.get("AUTO_STOCK_FILINGS", DATA / "stocks" / "sec_filings_raw.csv"))
STOCK_DATA = Path(os.environ.get("AUTO_STOCK_DATA", DATA / "stocks"))
STOCK_STATUS = Path(os.environ.get("AUTO_STOCK_STATUS", STOCK_DATA / "data_status.json"))
STOCK_DATA_MAX_AGE = max(120, int(os.environ.get("STOCK_DATA_MAX_AGE", "900")))
BAR_MS = 15 * 60 * 1000
CONFIG = os.environ.get("AUTO_CONFIG", "c")
TAIL = float(os.environ.get("AUTO_TAIL", "0.01"))
SIZE = os.environ.get("AUTO_SIZE", "0.01")
MAX_HOLD_BARS = int(os.environ.get("AUTO_MAX_HOLD_BARS", "48"))
INTERVAL = max(15, int(os.environ.get("AUTO_INTERVAL", "30")))
MAX_DATA_AGE = int(os.environ.get("AUTO_MAX_DATA_AGE", "1800"))
STATE_PATH = Path(os.environ.get("AUTO_STATE", DATA / "auto_demo_state.json"))
MAX_POSITIONS = max(1, int(os.environ.get("AUTO_MAX_POSITIONS", "2")))
STOCK_SIZE = os.environ.get("AUTO_STOCK_SIZE", "1")
STOCK_LOOKBACK_HOURS = max(1.0, float(os.environ.get("AUTO_STOCK_LOOKBACK_HOURS", "48")))
STOCK_STOP_BPS = float(os.environ.get("AUTO_STOCK_STOP_BPS", "600"))
STOCK_MAX_HOLD_HOURS = float(os.environ.get("AUTO_STOCK_MAX_HOLD_HOURS", "30"))
_last_stock_feed_warning = 0.0


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
        for col in ("sum_open_interest", "sum_open_interest_value", "count_long_short_ratio"):
            if col in oi:
                oi[col] = pd.to_numeric(oi[col], errors="coerce")
            if col in ratio:
                ratio[col] = pd.to_numeric(ratio[col], errors="coerce")
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
        return m


def load_state() -> dict:
    if not STATE_PATH.exists():
        return {"positions": {}, "last_bar": 0, "trades": 0}
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"positions": {}, "last_bar": 0, "trades": 0}


def save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=True, indent=2), encoding="utf-8")
    tmp.replace(STATE_PATH)


def remote_positions(client: DemoClient) -> dict[str, dict]:
    out = {}
    rows = []
    for inst_type in ("SWAP", "MARGIN"):
        try:
            rows.extend(client.positions(inst_type))
        except RuntimeError:
            # Spot tokenized stocks are represented by the margin account on
            # accounts where shorting is enabled; some accounts reject MARGIN.
            continue
    for row in rows:
        inst = row.get("instId", "")
        if not (inst.endswith("-USDT-SWAP") or inst.endswith("-USDT")):
            continue
        pos = float(row.get("pos") or 0)
        if abs(pos) < 1e-12:
            continue
        sym = inst.replace("-USDT-SWAP", "USDT") if inst.endswith("-USDT-SWAP") else inst
        out[sym] = {"inst_id": inst, "side": "long" if pos > 0 else "short",
                    "size": str(abs(pos)), "entry_px": float(row.get("avgPx") or 0)}
    return out


def stock_short_available(client: DemoClient, inst_id: str) -> bool:
    """Check the account's live borrow capacity before opening a stock short."""
    try:
        rows = client.max_loan(inst_id, mgn_ccy="USDT", mgn_mode="cross")
    except (requests.RequestException, RuntimeError, ValueError) as exc:
        print(f"stock short check failed {inst_id}: {exc}", flush=True)
        return False
    sell = next((row for row in rows if row.get("side") == "sell"), {})
    try:
        available = float(sell.get("maxLoan") or 0.0)
    except (TypeError, ValueError):
        available = 0.0
    if available <= 0:
        print(f"stock short skipped {inst_id}: account maxLoan(sell)={sell.get('maxLoan', '0')}", flush=True)
        return False
    return True


def build_crypto_signals(data: LiveData) -> dict[str, dict]:
    raw_k, raw_m, frames = {}, {}, {}
    for sym in SYMBOLS:
        raw_k[sym], raw_m[sym] = data.klines(sym), data.metrics(sym)
    newest = max(int(k.ts.max()) for k in raw_k.values())
    age = time.time() - newest / 1000.0
    if age > MAX_DATA_AGE:
        raise RuntimeError(f"stale OKX kline data: newest={newest} age={age:.0f}s")
    newest_metrics = max(int(m.ts.max()) for m in raw_m.values() if not m.empty)
    metrics_age = time.time() - newest_metrics / 1000.0
    if metrics_age > MAX_DATA_AGE:
        raise RuntimeError(f"stale OKX metrics data: newest={newest_metrics} age={metrics_age:.0f}s")
    btc_close = raw_k["BTCUSDT"].set_index("ts")["close"].astype(float)
    btc_ret = pd.Series(np.log(btc_close).diff(16).to_numpy(), index=btc_close.index)
    for sym in SYMBOLS:
        frames[sym] = build.build_symbol(sym, btc_ret if sym != "BTCUSDT" else None,
                                         klines_df=raw_k[sym], metrics_df=raw_m[sym])

    model = lgb.Booster(model_file=str(MODELS / f"dir_{CONFIG}_fold5.txt"))
    feats = model.feature_name()
    oos = pd.read_csv(RESULTS / f"oos_dir_{CONFIG}.csv.gz")
    last_fold = oos[oos.fold == oos.fold.max()]
    hi = float(last_fold[f"hi_{TAIL}"].iloc[-1])
    lo = float(last_fold[f"lo_{TAIL}"].iloc[-1])
    signals = {}
    cutoff = int(time.time() * 1000) // BAR_MS * BAR_MS
    for i, sym in enumerate(SYMBOLS):
        f = frames[sym]
        # LightGBM can route missing values. Requiring every feature to be
        # non-null would discard all OKX bars because OKX has no top-trader and
        # taker-ratio history; those fields are intentionally left as NaN.
        rows = f[f.ts < cutoff].dropna(subset=["sigma", "ret_96", "dist_ema_96", "dist_ema_384"])
        if rows.empty:
            continue
        row = rows.iloc[-1]
        x = row[[name for name in feats if name != "sym_code"]].to_frame().T.copy()
        x["sym_code"] = i
        x = x[feats]
        x = x.apply(pd.to_numeric, errors="coerce").astype(float)
        p = float(model.predict(x, num_iteration=model.current_iteration())[0])
        side = "long" if p >= hi else ("short" if p <= lo else "flat")
        close = float(raw_k[sym].set_index("ts").loc[int(row["ts"]), "close"])
        signals[sym] = {"bar": int(row["ts"]), "side": side, "p_up": p,
                        "close": close, "width": float(row[f"width_{CONFIG}" ]),
                        "strategy": "okx_quant_c", "asset_type": "crypto",
                        "inst_id": sym.replace("USDT", "-USDT-SWAP")}
    if signals:
        newest_signal = max(x["bar"] for x in signals.values())
        signal_age = time.time() - newest_signal / 1000.0
        if signal_age > MAX_DATA_AGE:
            raise RuntimeError(f"stale aligned feature data: newest={newest_signal} age={signal_age:.0f}s")
    return signals


def build_stock_signals(now: pd.Timestamp) -> dict[str, dict]:
    """Build tokenized-stock event signals from the locally refreshed SEC feed.

    The SEC feed is intentionally an input file: a separate poller can refresh
    it without coupling credentials or network retries to the trading loop.
    Events are restricted to the recent lookback so a restart cannot replay old
    historical backtest events as live orders.
    """
    global _last_stock_feed_warning
    if not STOCK_FILINGS.exists():
        return {}
    # A mounted CSV can survive a restarted poller indefinitely. Require a
    # recent successful heartbeat before treating it as a live feed; otherwise
    # the crypto sleeve can continue while stale stock events are skipped.
    try:
        status = json.loads(STOCK_STATUS.read_text(encoding="utf-8"))
        now_epoch = now.timestamp()
        updated = pd.Timestamp(status.get("updated_at"), tz="UTC").timestamp()
        sec_ok = pd.Timestamp(status.get("sec", {}).get("last_success_at"), tz="UTC").timestamp()
        candles_ok = pd.Timestamp(status.get("candles", {}).get("last_success_at"), tz="UTC").timestamp()
        if min(now_epoch - updated, now_epoch - sec_ok, now_epoch - candles_ok) < -5 or \
                max(now_epoch - updated, now_epoch - sec_ok, now_epoch - candles_ok) > STOCK_DATA_MAX_AGE:
            raise ValueError("stock feed heartbeat is stale")
    except (OSError, ValueError, TypeError, KeyError):
        if time.time() - _last_stock_feed_warning > STOCK_DATA_MAX_AGE:
            print(f"stock feed unavailable/stale: {STOCK_STATUS}", flush=True)
            _last_stock_feed_warning = time.time()
        return {}
    try:
        filings = pd.read_csv(STOCK_FILINGS)
        # SEC history contains several timestamp shapes (date-only legacy
        # rows and ISO rows with/without fractional seconds). Pandas 3's
        # strict single-format inference would silently drop the newest rows.
        filings["accepted"] = pd.to_datetime(filings["accepted"], utc=True,
                                              errors="coerce", format="mixed")
        cutoff = now - pd.Timedelta(hours=STOCK_LOOKBACK_HOURS)
        filings = filings.loc[
            (filings["accepted"] >= cutoff)
            & filings.form.astype(str).str.startswith("8-K")
            & filings.ticker.astype(str).isin(TECH)
        ].drop_duplicates("accession")
        if filings.empty:
            return {}
        inst_ids = sorted(filings.instId.dropna().unique())
        frames = stock_data.to_bar_end(stock_data.load_panel(inst_ids, "5m", STOCK_DATA), "5m")
        normal = news_strategy.build_events(
            filings, frames, observe_minutes=90, min_move_bps=50,
            max_stale_minutes=90, require_closed=True,
        )
        earnings = news_strategy.build_events(
            filings, frames, observe_minutes=45, min_move_bps=0,
            max_stale_minutes=90, require_closed=True,
        )
        universe_path = STOCK_FILINGS.parent / "universe.csv"
        if universe_path.exists():
            universe = pd.read_csv(universe_path)
            normal = news_strategy.annotate_shortable(normal, universe)
            earnings = news_strategy.annotate_shortable(earnings, universe)
        events = pd.concat([normal, earnings], ignore_index=True, sort=False)
        if events.empty:
            return {}
        out = {}
        for row in events.sort_values("event_ts").itertuples(index=False):
            # One signal per instrument; the latest event wins if filings overlap.
            if not bool(getattr(row, "executable", True)):
                continue
            event_ts = pd.Timestamp(row.event_ts)
            # Never enter a position on a stale event after a restart. The
            # event must be newly actionable within one loop window; older
            # events remain useful in the backtest but are not live signals.
            if event_ts < now - pd.Timedelta(minutes=max(30, INTERVAL * 2)) or event_ts > now:
                continue
            inst = str(row.inst_id)
            out[inst] = {
                "bar": int(event_ts.timestamp() * 1000),
                "side": "long" if int(row.side) > 0 else "short",
                "p_up": None, "close": None, "width": STOCK_STOP_BPS / 1e4,
                "strategy": "xstock_hybrid", "asset_type": "stock", "inst_id": inst,
                "event_ts": event_ts.isoformat(),
                "ticker": str(row.ticker), "resolve_ts": pd.Timestamp(row.resolve_ts).isoformat(),
                "observe_move_bps": float(row.abs_move_bps),
            }
        return out
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"stock signal build failed: {exc}", flush=True)
        return {}


def run_once(client: DemoClient, state: dict, allow_orders: bool) -> dict:
    data = LiveData(client.s)
    now = pd.Timestamp.now(tz="UTC")
    signals = build_crypto_signals(data)
    signals.update(build_stock_signals(now))
    # Continue managing a live position after its original stock event falls
    # outside the signal lookback window.
    for sym, pos in state.get("positions", {}).items():
        if sym not in signals:
            signals[sym] = {
                "bar": int(now.timestamp() * 1000) // BAR_MS * BAR_MS,
                "side": pos.get("side", "flat"), "p_up": None, "close": None,
                "width": float(pos.get("width") or STOCK_STOP_BPS / 1e4),
                "strategy": pos.get("strategy", "xstock_hybrid"),
                "asset_type": pos.get("asset_type", "stock"),
                "inst_id": pos.get("inst_id", sym),
            }
    # A closed-bar signal is deliberately separate from the live execution
    # price. Stops and timeouts must react to the OKX market now, not to the
    # close of the previous 15m bar (which can be minutes old).
    ticker_symbols = list(SYMBOLS) + [s for s, x in signals.items() if x.get("asset_type") == "stock"]
    tickers = {sym: data.ticker(sym, signals.get(sym, {}).get("asset_type", "crypto")) for sym in ticker_symbols}
    now_ms = int(time.time() * 1000)
    for sym, (_, ts) in tickers.items():
        if now_ms - ts > MAX_DATA_AGE * 1000:
            raise RuntimeError(f"stale OKX ticker data for {sym}: age={(now_ms - ts) / 1000:.0f}s")
    remote = remote_positions(client)
    last_bars = state.setdefault("last_bars", {})
    balance = client.balance()
    save_balance(balance)
    total_eq = balance[0].get("totalEq") if balance else None
    cycle_id = uuid.uuid4().hex
    # Reconcile remote state before considering a new signal.
    for sym in list(state["positions"]):
        if sym not in remote:
            state["positions"].pop(sym, None)
    for sym, pos in remote.items():
        state["positions"].setdefault(sym, {**pos, "opened_bar": 0, "width": 0.0})

    for sym, sig in signals.items():
        pos = state["positions"].get(sym)
        inst = sig.get("inst_id") or sym.replace("USDT", "-USDT-SWAP")
        if pos:
            bars = max(0, (sig["bar"] - int(pos.get("opened_bar") or sig["bar"])) // BAR_MS)
            entry, width = float(pos.get("entry_px") or sig.get("close") or tickers[sym][0]), float(pos.get("width") or sig["width"])
            mark = tickers[sym][0]
            stop = entry * (1 - width) if pos["side"] == "long" else entry * (1 + width)
            take = entry * (1 + width) if pos["side"] == "long" else entry * (1 - width)
            hit = (mark <= stop or mark >= take) if pos["side"] == "long" else (mark >= stop or mark <= take)
            max_bars = MAX_HOLD_BARS if sig.get("asset_type") == "crypto" else int(STOCK_MAX_HOLD_HOURS * 4)
            if allow_orders and (hit or bars >= max_bars):
                side = "sell" if pos["side"] == "long" else "buy"
                asset_type = pos.get("asset_type", sig.get("asset_type"))
                td_mode = "isolated" if asset_type == "crypto" else (
                    "cross" if pos["side"] == "short" else "cash"
                )
                quick_mgn_type = None
                if asset_type == "stock" and pos["side"] == "short":
                    quick_mgn_type = "auto_repay" if pos["side"] == "short" else None
                client.order(inst, side, pos.get("size", SIZE), td_mode, True, quick_mgn_type)
                state["positions"].pop(sym, None)
                state["trades"] = int(state.get("trades", 0)) + 1
                print(f"EXIT {sym} strategy={pos.get('strategy', sig.get('strategy'))} side={pos['side']} mark={mark:.8g} bars={bars} hit={hit}", flush=True)
            continue
        if sig["side"] == "flat" or sig["bar"] <= int(last_bars.get(sym, 0)):
            continue
        if allow_orders and len(state["positions"]) < MAX_POSITIONS:
            side = "buy" if sig["side"] == "long" else "sell"
            size = SIZE if sig.get("asset_type") == "crypto" else STOCK_SIZE
            td_mode = "isolated" if sig.get("asset_type") == "crypto" else "cash"
            quick_mgn_type = None
            if sig.get("asset_type") == "stock":
                if sig["side"] == "short" and not stock_short_available(client, inst):
                    continue
                td_mode = "cross"
                quick_mgn_type = "auto_borrow" if sig["side"] == "short" else None
            result = client.order(inst, side, size, td_mode, False, quick_mgn_type)
            ord_id = result[0].get("ordId") if result else ""
            detail = client.order_detail(inst, ord_id) if ord_id else {}
            if detail.get("state") == "filled":
                state["positions"][sym] = {"inst_id": inst, "side": sig["side"], "size": size,
                                           "entry_px": float(detail.get("avgPx") or sig.get("close") or tickers[sym][0]),
                                           "width": sig["width"], "opened_bar": sig["bar"], "ord_id": ord_id,
                                           "strategy": sig.get("strategy"), "asset_type": sig.get("asset_type")}
                state["trades"] = int(state.get("trades", 0)) + 1
                p_text = f"{sig['p_up']:.5f}" if sig.get("p_up") is not None else "n/a"
                print(f"ENTRY {sym} strategy={sig.get('strategy')} side={sig['side']} p={p_text} ord={ord_id}", flush=True)
    for sym, sig in signals.items():
        last_bars[sym] = max(int(last_bars.get(sym, 0)), int(sig.get("bar") or 0))
    state["last_bar"] = max(last_bars.values(), default=int(state.get("last_bar", 0)))
    # Store a row for every symbol on every cycle, including flat symbols. This
    # makes exposure, signal drift, latency and mark-to-market history queryable
    # at the same cadence as the strategy loop.
    snapshot_rows = []
    for sym in ticker_symbols:
        sig = signals.get(sym, {})
        pos = state["positions"].get(sym, {})
        mark, ticker_ts = tickers[sym]
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
            "take_px": (entry * (1 + width) if side == "long" else entry * (1 - width)) if side and entry and width else None,
            "total_eq": total_eq,
            "raw_json": json.dumps({"signal": sig, "position": pos}, ensure_ascii=True),
        })
    save_strategy_snapshots(snapshot_rows)
    save_state(state)
    print("signals:", json.dumps(signals, ensure_ascii=True),
          "okx_tickers:", json.dumps({s: {"last": p, "ts": ts} for s, (p, ts) in tickers.items()},
                                      ensure_ascii=True),
          "positions:", len(state["positions"]), flush=True)
    return state


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--observe", action="store_true", help="never send orders")
    args = ap.parse_args()
    if CONFIG != "c" or abs(TAIL - 0.01) > 1e-9:
        print(f"warning: non-default strategy config={CONFIG} tail={TAIL}", flush=True)
    client = DemoClient()
    state = load_state()
    allow = os.environ.get("AUTO_TRADE", "false").lower() in {"1", "true", "yes", "on"} and not args.observe
    print(f"auto demo config={CONFIG} tail={TAIL} allow_orders={allow} interval={INTERVAL}s "
          f"max_positions={MAX_POSITIONS} max_data_age={MAX_DATA_AGE}s "
          f"price_source=OKX metrics_source=OKX simulated_header={int(SIMULATED_TRADING)}", flush=True)
    while True:
        try:
            run_once(client, state, allow)
        except (requests.RequestException, RuntimeError, ValueError, KeyError) as exc:
            print(f"auto cycle failed: {exc}", flush=True)
        if args.once:
            return
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
