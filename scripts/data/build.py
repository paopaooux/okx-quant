"""Build the 15m feature/label panel for BTC, ETH, SOL (Binance USDT-M perp).

Two rules govern everything here:

1.  No lookahead.  Every feature at bar t is computable from information
    available at the *close* of bar t.  Derivatives metrics get an extra one-bar
    lag on top of that, because the 15m fold takes the last 5m snapshot inside
    the bar rather than at its edge.

2.  The label is the trade, not a return.  A triple barrier (take-profit /
    stop-loss / time-out) asks the question the objective actually poses -- "if
    I open here, do I get paid before I get stopped, and soon?" -- rather than
    "what is the mean forward return", which a high-win-rate mandate does not
    care about.  Entry is the *next* bar's open, so the decision bar is never
    part of the outcome.

Barriers are symmetric in volatility units, which pins the coin-flip base rate
near (1 - P(timeout))/2 and makes any lift above it interpretable.  An
asymmetric barrier would manufacture a high win rate with no edge at all.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
DEFAULT_SYMBOLS = (
    "ADAUSDT", "BNBUSDT", "BTCUSDT", "DOGEUSDT",
    "ETHUSDT", "LINKUSDT", "SOLUSDT", "XRPUSDT",
)
_symbols_arg = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--symbols=")), None)
SYMBOLS = tuple(s.strip().upper() for s in _symbols_arg.split(",") if s.strip()) if _symbols_arg else DEFAULT_SYMBOLS
_output_arg = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--output=")), None)

# Binance USDT-M perp, VIP0: 4bps taker per side, plus ~1bp of spread/slippage
# on majors.  Do NOT reuse the OKX-spot cost model from ../distilled_alpha --
# that one carries a 25bps half-spread calibrated for microcaps and would make
# every perp result look impossible.
ROUND_TRIP_BPS = 10.0

BAR_MIN = 15
VOL_SPAN = 96   # 24h EWM for the volatility estimate

# (name, K, H): barrier half-width is K * sigma_bar * sqrt(H), so K is measured
# against the *terminal* dispersion of the holding period, not a single bar.
#
# The reason to carry a grid rather than one choice: the round-trip cost is a
# fixed ~10bps, so a tight barrier demands an implausible win rate to break even
# (at 4.8*sigma ~ 55bps the break-even is 59%) while a wide one demands less but
# holds longer.  The objective asks for short holds AND high win rate, and those
# pull against each other through the cost floor.  Report all three; do not
# quietly pick the flattering one.
CONFIGS = (("a", 1.2, 16),    # 4h  -- shortest hold
           ("b", 1.5, 32),    # 8h
           ("c", 2.0, 48))    # 12h


# ---------------------------------------------------------------- labels

def triple_barrier(open_: np.ndarray, high: np.ndarray, low: np.ndarray,
                   width: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """First-touch outcome for a trade opened at the next bar's open.

    Returns (long_win, short_win, bars_held, exit_ret_long) where exit_ret_long
    is the realised gross return of the long trade at whichever barrier fired.

    When a single bar's range spans both barriers the touch order is unknowable
    from OHLC, so it is scored as a *loss* for whichever side is being asked
    about.  That is the pessimistic reading and it keeps the win rate honest.
    """
    n = len(open_)
    lw = np.zeros(n, dtype=np.int8)
    sw = np.zeros(n, dtype=np.int8)
    held = np.full(n, np.nan)
    ret = np.full(n, np.nan)

    for i in range(n - horizon - 1):
        w = width[i]
        if not np.isfinite(w) or w <= 0:
            continue
        entry = open_[i + 1]
        if not np.isfinite(entry) or entry <= 0:
            continue
        up = entry * (1.0 + w)
        dn = entry * (1.0 - w)
        end = i + 1 + horizon
        hit_up = hit_dn = -1
        for j in range(i + 1, end + 1):
            u = high[j] >= up
            d = low[j] <= dn
            if u and d:                      # ambiguous bar -> both sides lose
                hit_up = hit_dn = j
                break
            if u:
                hit_up = j
                break
            if d:
                hit_dn = j
                break
        if hit_up >= 0 and hit_dn >= 0:      # ambiguous
            held[i] = hit_up - i
            ret[i] = 0.0
        elif hit_up >= 0:
            lw[i] = 1
            held[i] = hit_up - i
            ret[i] = w
        elif hit_dn >= 0:
            sw[i] = 1
            held[i] = hit_dn - i
            ret[i] = -w
        else:                                # timed out
            held[i] = horizon
            ret[i] = open_[end] / entry - 1.0
    tail = n - horizon - 1
    lw[tail:] = 0
    sw[tail:] = 0
    held[tail:] = np.nan
    ret[tail:] = np.nan
    return lw, sw, held, ret


# ---------------------------------------------------------------- features

def zscore(s: pd.Series, w: int) -> pd.Series:
    """Rolling z-score using moments that exclude the current observation."""
    m = s.rolling(w, min_periods=w // 2).mean().shift(1)
    sd = s.rolling(w, min_periods=w // 2).std().shift(1)
    return (s - m) / sd.replace(0.0, np.nan)


def rsi(close: pd.Series, w: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / w, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / w, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0.0, np.nan))


def build_symbol(sym: str, btc_ret: pd.Series | None, limit: int | None = None,
                 klines_df: pd.DataFrame | None = None,
                 metrics_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build features from local archives or caller-provided live frames."""
    k = (klines_df.copy() if klines_df is not None else
         pd.read_csv(DATA / "klines" / f"{sym}.csv.gz"))
    m = (metrics_df.copy() if metrics_df is not None else
         pd.read_csv(DATA / "metrics" / f"{sym}.csv.gz"))

    df = k.merge(m, on="ts", how="left").sort_values("ts").reset_index(drop=True)
    if limit is not None:                 # used by selftest.py to prove no lookahead
        df = df.iloc[:limit].reset_index(drop=True)
    df["dt"] = pd.to_datetime(df["ts"], unit="ms", utc=True)

    # The metrics fold takes the last 5m snapshot *inside* the bar, which lands
    # before the bar closes -- no lookahead as such -- but one extra bar of lag
    # removes any doubt and costs one bar of freshness.
    mcols = ["sum_open_interest", "sum_open_interest_value",
             "count_toptrader_long_short_ratio", "sum_toptrader_long_short_ratio",
             "count_long_short_ratio", "sum_taker_long_short_vol_ratio"]
    for c in mcols:
        df[c] = pd.to_numeric(df.get(c), errors="coerce").shift(1)

    c, h, l, o = df["close"], df["high"], df["low"], df["open"]
    logc = np.log(c)
    r1 = logc.diff()

    f = pd.DataFrame(index=df.index)
    f["symbol"] = sym
    f["ts"] = df["ts"]
    f["dt"] = df["dt"]

    # --- price / momentum -------------------------------------------------
    for n in (1, 4, 16, 96):
        f[f"ret_{n}"] = logc.diff(n)
    f["ret_16_x_96"] = np.sign(f["ret_96"]) * f["ret_16"]      # with- or against-trend push

    sigma = r1.ewm(span=VOL_SPAN, adjust=False).std().shift(1)
    f["sigma"] = sigma
    f["vol_ratio_4_96"] = (r1.ewm(span=16, adjust=False).std().shift(1) / sigma)
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    f["atr_pct"] = (tr.ewm(span=14, adjust=False).mean() / c)
    f["vol_z"] = zscore(sigma, 384)

    rng = (h.rolling(96).max() - l.rolling(96).min())
    f["range_pos_96"] = (c - l.rolling(96).min()) / rng.replace(0.0, np.nan)
    f["dd_from_high_96"] = c / h.rolling(96).max() - 1.0
    f["dist_ema_96"] = c / c.ewm(span=96, adjust=False).mean() - 1.0
    f["dist_ema_384"] = c / c.ewm(span=384, adjust=False).mean() - 1.0
    f["rsi_14"] = rsi(c, 14)
    bar_rng = (h - l).replace(0.0, np.nan)
    f["upper_wick"] = (h - np.maximum(o, c)) / bar_rng
    f["lower_wick"] = (np.minimum(o, c) - l) / bar_rng
    f["body"] = (c - o) / bar_rng

    # --- volume / order flow ---------------------------------------------
    qv = df["quote_volume"]
    f["volume_z_96"] = zscore(qv, 96)
    f["volume_ratio_16_96"] = qv.rolling(16).mean() / qv.rolling(96).mean().replace(0.0, np.nan)
    f["trade_size_z"] = zscore(qv / df["count"].replace(0, np.nan), 96)
    tbr = (df["taker_buy_quote_volume"] / qv.replace(0.0, np.nan))
    f["taker_buy_ratio"] = tbr
    f["tbr_mean_16"] = tbr.rolling(16).mean()
    f["tbr_z_96"] = zscore(tbr, 96)

    # --- open interest ----------------------------------------------------
    oi = df["sum_open_interest"]
    logoi = np.log(oi.replace(0.0, np.nan))
    for n in (1, 4, 16, 96):
        f[f"oi_chg_{n}"] = logoi.diff(n)
    f["oi_z_96"] = zscore(oi, 96)
    f["oi_z_384"] = zscore(oi, 384)
    # The textbook read: OI rising *with* price is new money taking the trend on;
    # OI falling with price rising is short covering, which has no one left to
    # push it further.  Signing the OI change by the price move separates them.
    f["oi_price_div_16"] = np.sign(f["ret_16"]) * f["oi_chg_16"]
    f["oi_price_div_4"] = np.sign(f["ret_4"]) * f["oi_chg_4"]
    f["oi_ret_corr_96"] = f["oi_chg_1"].rolling(96).corr(f["ret_1"])
    f["oi_per_price"] = f["oi_chg_16"] - f["ret_16"]           # leverage build vs price alone

    # --- positioning ------------------------------------------------------
    tt_pos = df["sum_toptrader_long_short_ratio"]
    tt_acc = df["count_toptrader_long_short_ratio"]
    glob = df["count_long_short_ratio"]
    taker = df["sum_taker_long_short_vol_ratio"]
    for name, s in (("tt_pos", tt_pos), ("tt_acc", tt_acc), ("glob", glob), ("taker", taker)):
        f[f"{name}"] = s
        f[f"{name}_z_96"] = zscore(s, 96)
        f[f"{name}_chg_16"] = s.diff(16)
    # Top traders leaning the opposite way from the retail crowd is the whole
    # reason both series are published; the spread is the usable form.
    f["crowding"] = np.log(tt_pos.replace(0, np.nan)) - np.log(glob.replace(0, np.nan))
    f["crowding_z_96"] = zscore(f["crowding"], 96)
    f["taker_mean_16"] = taker.rolling(16).mean()

    # --- market factor ----------------------------------------------------
    if btc_ret is not None:
        b = btc_ret.reindex(df["ts"]).to_numpy()
        f["btc_ret_16"] = b
        f["rel_strength_16"] = f["ret_16"] - f["btc_ret_16"]
        f["beta_96"] = f["ret_1"].rolling(96).cov(pd.Series(b, index=f.index)) / \
            pd.Series(b, index=f.index).rolling(96).var().replace(0.0, np.nan)
    else:
        f["btc_ret_16"] = f["ret_16"]
        f["rel_strength_16"] = 0.0
        f["beta_96"] = 1.0

    # --- session ----------------------------------------------------------
    f["hour"] = df["dt"].dt.hour.astype("int16")
    f["dow"] = df["dt"].dt.dayofweek.astype("int16")

    # --- labels -----------------------------------------------------------
    oa, ha, la = o.to_numpy(float), h.to_numpy(float), l.to_numpy(float)
    for name, kk, hh in CONFIGS:
        width = (kk * sigma * np.sqrt(hh)).to_numpy()
        lw, sw, held, ret = triple_barrier(oa, ha, la, width, hh)
        f[f"width_{name}"] = width
        f[f"long_win_{name}"] = lw
        f[f"short_win_{name}"] = sw
        f[f"held_{name}"] = held
        f[f"exit_ret_{name}"] = ret
    f["entry_px"] = o.shift(-1)
    return f


def main() -> None:
    btc = pd.read_csv(DATA / "klines" / "BTCUSDT.csv.gz")[["ts", "close"]]
    btc_ret = pd.Series(np.log(btc["close"]).diff(16).to_numpy(), index=btc["ts"].to_numpy())

    frames = []
    for sym in SYMBOLS:
        f = build_symbol(sym, btc_ret if sym != "BTCUSDT" else None)
        need = ["sigma", "oi_chg_96", "ret_96", "entry_px"] + \
               [f"long_win_{n}" for n, _, _ in CONFIGS]
        f = f.dropna(subset=need)
        frames.append(f)
        print(f"{sym}: {len(f):,} bars  {f['dt'].min():%Y-%m-%d} -> {f['dt'].max():%Y-%m-%d}")
        for name, kk, hh in CONFIGS:
            bl, bs = f[f"long_win_{name}"].mean(), f[f"short_win_{name}"].mean()
            w = f[f"width_{name}"].median()
            # Break-even win rate for a symmetric 1:1 barrier at ROUND_TRIP_BPS cost,
            # ignoring time-outs: p* = 0.5 + cost / (2 * width).
            be = 0.5 + (ROUND_TRIP_BPS / 1e4) / (2 * w)
            print(f"   {name} K={kk} H={hh}: width {w*1e4:5.0f}bps  "
                  f"long {bl:.1%} short {bs:.1%} timeout {1-bl-bs:.1%}  "
                  f"median hold {f[f'held_{name}'].median():.0f} bars  "
                  f"break-even winrate {be:.1%}", flush=True)

    panel = pd.concat(frames, ignore_index=True).sort_values(["ts", "symbol"]).reset_index(drop=True)
    out = Path(_output_arg) if _output_arg else DATA / "panel.csv.gz"
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(out, index=False, compression="gzip")
    print(f"\npanel: {len(panel):,} rows x {panel.shape[1]} cols -> {out}")


if __name__ == "__main__":
    main()
