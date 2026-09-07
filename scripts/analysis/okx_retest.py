"""Re-test the frozen crypto sleeve on a downloaded OKX archive.

This separates two questions: (1) keep the old Binance model signal and only
change execution prices/labels to OKX for the whole interval; (2) where OKX
15-minute trading statistics exist, run the old model on OKX-native inputs.
The latter is a portability study, not a production model replacement.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from scripts.combinations.run import QUANT_SYMBOLS, _pooled_curve
from scripts.data import build
from strategies.stocks.market import data as stock_data
from strategies.stocks.market import sessions
from strategies.stocks.config import Config
from strategies.stocks.research import events
from strategies.stocks.research.backtest import Rules, run as stock_backtest
from scripts.analysis.okx_native_study import coverage

STEP = pd.Timedelta(minutes=15)
START = pd.Timestamp("2026-03-03T09:30:00Z")
END = pd.Timestamp("2026-09-03T14:30:00Z")


def load_okx(root: Path):
    out = {}
    for symbol in QUANT_SYMBOLS:
        frame = pd.read_csv(root / "klines" / f"{symbol}.csv.gz")
        frame["ts"] = pd.to_datetime(frame.ts, unit="ms", utc=True)
        frame = frame.drop_duplicates("ts").sort_values("ts").set_index("ts")
        out[symbol] = frame
    return out


def _barrier_candidates(oos, okx):
    """Old OOS direction signals, with OKX OHLC barrier outcomes."""
    rows = []
    for symbol in QUANT_SYMBOLS:
        frame = okx[symbol]
        group = oos.loc[(oos.symbol == symbol) & (oos.dt >= frame.index.min()) & (oos.dt <= END)].sort_values("ts")
        group = group.loc[(group.p_up >= group["hi_0.01"]) | (group.p_up <= group["lo_0.01"])]
        free_at = frame.index.min()
        for i, r in group.iterrows():
            stamp = r["dt"]
            if stamp < free_at or not np.isfinite(r.held_c) or not np.isfinite(r.width_c) or r.width_c <= 0:
                continue
            side = 1 if r.p_up >= r["hi_0.01"] else -1 if r.p_up <= r["lo_0.01"] else 0
            if not side:
                continue
            entry_ts = stamp + STEP
            entry_i = frame.index.searchsorted(entry_ts, side="left")
            end_i = entry_i + 48
            if entry_i >= len(frame) or end_i >= len(frame) or frame.index[entry_i] != entry_ts:
                continue
            if frame.index[end_i] != entry_ts + 48 * STEP:
                raise ValueError(f"Noncontiguous OKX execution path for {symbol} {stamp}")
            entry = float(frame.open.iloc[entry_i])
            width = float(r.width_c)
            up, dn = entry * (1 + width), entry * (1 - width)
            exit_i, ret = None, None
            for j in range(entry_i, end_i + 1):
                hi, lo = float(frame.high.iloc[j]), float(frame.low.iloc[j])
                hit_up, hit_dn = hi >= up, lo <= dn
                if hit_up or hit_dn:
                    exit_i = j
                    ret = 0.0 if hit_up and hit_dn else width if hit_up else -width
                    break
            timeout = exit_i is None
            if timeout:
                exit_i, ret = end_i, float(frame.open.iloc[end_i] / entry - 1)
            exit_ts = frame.index[exit_i] + (pd.Timedelta(0) if timeout else STEP)
            rows.append(dict(strategy="okx_quant_c_tail_0.01", symbol=symbol,
                             entry_ts=stamp, actual_entry_ts=entry_ts, exit_ts=exit_ts,
                             side=side, gross=side * ret,
                             signal_strength=abs(float(r.p_up) - .5), source_i=i,
                             hold_hours=(exit_ts - stamp).total_seconds() / 3600,
                             held_okx=48 if timeout else exit_i - entry_i + 1, width=float(width)))
            free_at = exit_ts
    return pd.DataFrame(rows)


def _native_metrics(root: Path, symbol: str, candle: pd.DataFrame):
    """Join downloaded OKX Rubik fields to bars without filling history."""
    fields = {
        "oi": "sum_open_interest", "top_account": "count_toptrader_long_short_ratio",
        "top_position": "sum_toptrader_long_short_ratio", "global": "count_long_short_ratio",
    }
    metrics = pd.DataFrame(index=candle.index)
    for source, target in fields.items():
        path = root / "metrics" / f"{symbol}_{source}_15m.csv.gz"
        if not path.exists():
            continue
        raw = pd.read_csv(path)
        raw["ts"] = pd.to_datetime(raw.ts, unit="ms", utc=True)
        raw = raw.drop_duplicates("ts").set_index("ts").sort_index()
        value = "oi_ccy" if source == "oi" and "oi_ccy" in raw else target
        if value not in raw:
            continue
        metrics[target] = pd.to_numeric(raw[value], errors="coerce").reindex(candle.index)
    taker_path = root / "metrics" / f"{symbol}_taker_15m.csv.gz"
    if taker_path.exists():
        raw = pd.read_csv(taker_path)
        raw["ts"] = pd.to_datetime(raw.ts, unit="ms", utc=True)
        raw = raw.drop_duplicates("ts").set_index("ts").sort_index()
        sell = pd.to_numeric(raw.sell_quote_volume, errors="coerce")
        buy = pd.to_numeric(raw.buy_quote_volume, errors="coerce")
        metrics["sum_taker_long_short_vol_ratio"] = (buy / sell.replace(0, np.nan)).reindex(candle.index)
        share = (buy / (buy + sell).replace(0, np.nan)).reindex(candle.index)
    else:
        share = pd.Series(np.nan, index=candle.index)
    k = candle.copy()
    k["ts"] = [int(t.timestamp() * 1000) for t in k.index]
    k["quote_volume"] = pd.to_numeric(k["volume_quote"], errors="coerce")
    k["count"] = np.nan
    k["taker_buy_quote_volume"] = k["quote_volume"] * share
    k["taker_buy_volume"] = k["volume"] * share
    for col in ("sum_open_interest", "sum_open_interest_value", "count_toptrader_long_short_ratio",
                "sum_toptrader_long_short_ratio", "count_long_short_ratio",
                "sum_taker_long_short_vol_ratio"):
        if col not in metrics:
            metrics[col] = np.nan
    metrics["ts"] = k["ts"].to_numpy()
    return k.reset_index(drop=True), metrics.reset_index(drop=True)


def _model_candidates(root: Path, start, end, entry_end, limited: bool = False):
    okx = load_okx(root)
    model = lgb.Booster(model_file="models/dir_c_roll730_fold5.txt")
    rows = []
    coverage = {}
    thresholds = pd.read_csv("results/crypto/oos_dir_c_roll730.csv.gz", usecols=["fold", "hi_0.01", "lo_0.01"])
    hi = float(thresholds.loc[thresholds.fold.eq(5), "hi_0.01"].iloc[0])
    lo = float(thresholds.loc[thresholds.fold.eq(5), "lo_0.01"].iloc[0])
    btc = okx["BTCUSDT"].close
    btc_ret = pd.Series(np.log(btc).diff(16).to_numpy(), index=[int(t.timestamp()*1000) for t in btc.index])
    for symbol in QUANT_SYMBOLS:
        candle = okx[symbol].loc[okx[symbol].index < end].copy()
        k, metrics = _native_metrics(root, symbol, candle)
        if limited:
            # Counterfactual matching the current adapter: these columns are
            # absent/unknown, while OHLC and the current neutral proxies stay.
            for col in ("count_toptrader_long_short_ratio", "sum_toptrader_long_short_ratio",
                        "sum_taker_long_short_vol_ratio"):
                metrics[col] = np.nan
            k["count"] = k["volume"]
            k["taker_buy_quote_volume"] = k["quote_volume"] * .5
            k["taker_buy_volume"] = k["volume"] * .5
        f = build.build_symbol(symbol, btc_ret if symbol != "BTCUSDT" else None, klines_df=k, metrics_df=metrics)
        f["sym_code"] = sorted(QUANT_SYMBOLS).index(symbol)
        need = [x for x in model.feature_name() if x != "sym_code"]
        x = f[need + ["sym_code"]].apply(pd.to_numeric, errors="coerce")[model.feature_name()]
        f["p_up"] = model.predict(x, num_iteration=model.current_iteration(), num_threads=2)
        sample_x = x.loc[(f.dt >= start) & (f.dt <= entry_end)]
        coverage[symbol] = dict(bars=len(f), first=str(f.dt.min()), last=str(f.dt.max()),
                                missing_features=int(x.iloc[-1].isna().sum()),
                                missing_names=x.columns[x.iloc[-1].isna()].tolist(),
                                sample_missing_max=int(sample_x.isna().sum(axis=1).max()),
                                sample_missing_fraction=sample_x.isna().mean().to_dict())
        group = f.loc[(f.dt >= start) & (f.dt <= entry_end)].copy()
        group["signal_side"] = np.where(group.p_up >= hi, 1, np.where(group.p_up <= lo, -1, 0))
        group = group.loc[group.signal_side != 0]
        group["entry_ts"] = group.dt
        group["actual_entry_ts"] = group.dt + STEP
        group["exit_ts"] = group.dt + (group.held_c + 1) * STEP
        group["gross"] = group.signal_side * group.exit_ret_c
        group["side"] = group.signal_side
        group["strategy"] = "okx_native_limited" if limited else "okx_native_full"
        group["symbol"] = symbol
        group["signal_strength"] = (group.p_up - .5).abs()
        group = group.loc[np.isfinite(group.gross) & (group.exit_ts <= end)]
        group["hold_hours"] = (group.exit_ts - group.entry_ts).dt.total_seconds() / 3600
        # Preserve one net position per symbol using the native label clock.
        accepted, free_at = [], -1
        for i, row in group.sort_values("entry_ts").iterrows():
            if int(i) < free_at:
                continue
            accepted.append(row.to_dict())
            free_at = int(i + row.held_c + 1)
        rows.extend(accepted)
        print(f"model {'limited' if limited else 'filled'} {symbol}: trades={len(accepted)}, missing={coverage[symbol]['missing_features']}", flush=True)
    columns = ["strategy", "symbol", "entry_ts", "exit_ts", "side", "gross", "signal_strength", "hold_hours"]
    return pd.DataFrame(rows).reindex(columns=columns), coverage


def recent_stocks(start, end, entry_end):
    universe = pd.read_csv("data/stocks_swap/universe.csv")
    frames = stock_data.to_bar_end(stock_data.load_panel(sorted(universe.instId), "5m", "data/stocks_swap"), "5m")
    frames = {k: v.loc[(v.index >= start - pd.Timedelta(days=18)) & (v.index <= end)] for k, v in frames.items()}
    frames = {k: v for k, v in frames.items() if len(v)}
    windows = sessions.closed_windows(start - pd.Timedelta(days=5), end + pd.Timedelta(days=8))
    e = events.off_hours_dislocation(frames, Config(dislocation_bps=600), windows=windows)
    e = e.loc[(e.event_ts >= start) & (e.event_ts <= entry_end)]
    rules = Rules(horizon="to_open", max_hold_hours=30, stop_loss_bps=300, take_profit_bps=0,
                  max_concurrent=3, max_per_day=2, rank_column="abs_deviation", resolve_offset_minutes=60,
                  min_trailing_volume=0)
    trades = stock_backtest(e, frames, Config(), rules).trades
    if trades.empty:
        return pd.DataFrame(columns=["strategy", "symbol", "entry_ts", "exit_ts", "gross", "net", "signal_strength"])
    trades = trades.loc[(trades.entry_ts >= start) & (trades.exit_ts <= end)].copy()
    return trades.assign(strategy="xstock_hybrid", symbol=trades.inst_id, signal_strength=trades.signal.abs())


def stats(trades, days):
    if trades.empty:
        return dict(trades=0, return_pct=0., max_dd_pct=0., win_pct=0., avg_net_bps=0.)
    curve, accepted, skipped = _pooled_curve(trades, days, 5)
    net = accepted.net
    return dict(trades=len(accepted), skipped=int(skipped),
                return_pct=float((curve.iloc[-1] - 1) * 100),
                max_dd_pct=float((curve / curve.cummax() - 1).min() * 100),
                win_pct=float((net > 0).mean() * 100), avg_net_bps=float(net.mean() * 1e4))


def compare_bundle(stock, crypto, days, output, name):
    """Preserve the frozen two-stage sleeve admission and shared capital pool."""
    crypto = crypto.copy()
    crypto["net"] = crypto.gross - .001
    _, crypto, _ = _pooled_curve(crypto, days, 5)
    combined = pd.concat([stock, crypto], ignore_index=True)
    curve, accepted, _ = _pooled_curve(combined, days, 5)
    crypto.to_csv(output / f"{name}_crypto_trades.csv", index=False)
    accepted.to_csv(output / f"{name}_combined_trades.csv", index=False)
    curve.rename("equity").to_csv(output / f"{name}_daily_equity.csv")
    result = dict(crypto=stats(crypto, days), combined=stats(combined, days))
    result["combined"]["stock_trades"] = int(accepted.strategy.eq("xstock_hybrid").sum())
    result["combined"]["crypto_trades"] = int(accepted.strategy.ne("xstock_hybrid").sum())
    return result


def correlations(okx, root):
    rows = []
    for symbol, candle in okx.items():
        bn = pd.read_csv(Path("data/klines") / f"{symbol}.csv.gz", usecols=["ts", "close"])
        bn.index = pd.to_datetime(bn.ts, unit="ms", utc=True)
        pair = pd.concat([bn.close.rename("binance"), candle.close.rename("okx")], axis=1)
        pair = pair.loc[(pair.index >= START) & (pair.index <= END)].dropna()
        returns = np.log(pair).diff().loc[pair.index.to_series().diff().eq(STEP)]
        taker = pd.read_csv(root / "metrics" / f"{symbol}_taker_15m.csv.gz")
        taker.index = pd.to_datetime(taker.ts, unit="ms", utc=True)
        total = taker.buy_quote_volume + taker.sell_quote_volume
        qv = candle.volume_quote.reindex(total.index)
        rows.append(dict(symbol=symbol, paired_bars=len(pair),
                         price_correlation=float(pair.binance.corr(pair.okx)),
                         return_15m_correlation=float(returns.binance.corr(returns.okx)),
                         taker_volume_same_bar_correlation=float(total.corr(qv)),
                         taker_volume_to_candle_median_ratio=float((total / qv).median())))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path("data/research_okx_native_20260907"))
    ap.add_argument("--output", type=Path, default=Path("results/okx_retest_20260907"))
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    oos = pd.read_csv("results/crypto/oos_dir_c_roll730.csv.gz")
    oos["dt"] = pd.to_datetime(oos.dt, utc=True)
    oos["ts"] = pd.to_numeric(oos.ts)
    okx = load_okx(args.data)
    old_signal = _barrier_candidates(oos, okx)
    old_signal = old_signal.loc[(old_signal.entry_ts >= START) & (old_signal.exit_ts <= END)].copy()
    old_signal["net"] = old_signal.gross - .001
    days = pd.date_range(START.normalize(), END.normalize(), freq="D")
    stock = pd.read_csv("results/combinations/latest/stock_trades.csv")
    quant = pd.read_csv("results/combinations/latest/quant_trades.csv")
    for frame in (stock, quant):
        for col in ("entry_ts", "exit_ts"):
            frame[col] = pd.to_datetime(frame[col], utc=True)
    stock = stock.assign(strategy="xstock_hybrid", symbol=stock.inst_id, signal_strength=stock.signal.abs())
    stock["net"] = stock.gross - .0044
    baseline = compare_bundle(stock, quant, days, args.output, "frozen_baseline")
    if not np.isclose(baseline["combined"]["return_pct"], 50.2771537515, atol=1e-7):
        raise ValueError(f"Frozen baseline failed reproduction: {baseline}")
    portability = compare_bundle(stock, old_signal, days, args.output, "old_signal_okx_price")
    print("Frozen window:", json.dumps(dict(baseline=baseline, portability=portability)), flush=True)
    manifest = json.loads((args.data / "download_manifest.json").read_text())
    if not manifest.get("complete"):
        raise ValueError("Download must finish before running the study")
    recent_start = max(pd.Timestamp(c["start"]) for c in manifest["metrics"].values()) + 386 * STEP
    recent_end = pd.Timestamp(manifest["end"])
    entry_end = recent_end - pd.Timedelta(hours=30)
    recent_days = pd.date_range(recent_start.normalize(), recent_end.normalize(), freq="D")
    recent_stock = recent_stocks(recent_start, recent_end, entry_end)
    recent_stock.to_csv(args.output / "recent_stock_trades.csv", index=False)
    native, full_cov = _model_candidates(args.data, recent_start, recent_end, entry_end, limited=False)
    limited, limited_cov = _model_candidates(args.data, recent_start, recent_end, entry_end, limited=True)
    native_result = compare_bundle(recent_stock, native, recent_days, args.output, "recent_filled")
    limited_result = compare_bundle(recent_stock, limited, recent_days, args.output, "recent_masked")
    result = dict(
        downloaded_data=str(args.data), interval_start=str(START), interval_end=str(END),
        frozen_baseline=baseline, price_portability=portability,
        recent_start=str(recent_start), recent_entry_cutoff=str(entry_end), recent_end=str(recent_end),
        native_recent=native_result, masked_recent=limited_result,
        market_correlations=correlations(okx, args.data),
        coverage_full=full_cov, coverage_limited=limited_cov,
        source_hash=hashlib.sha256((args.data / "download_manifest.json").read_bytes()).hexdigest(),
        model_sha256=hashlib.sha256(Path("models/dir_c_roll730_fold5.txt").read_bytes()).hexdigest(),
        caveats=[
            "Price portability keeps the frozen Binance model scores and only replaces OHLC outcomes with OKX candles.",
            "Native runs reuse the old Binance-trained model without refitting; OKX 15m statistics are only available for the recent API retention window.",
            "native_limited is a same-price masked-input counterfactual with full recent OI/global data, NOT an exact replay of the current live sparse OI/global adapter.",
            "Taker-volume ratios are mapped from the documented OKX buy/sell fields; trade count is unavailable and remains missing.",
            "No native OKX model is deployed. This study does not change the live data source or strategy.",
            "Fixed assumed round-trip costs: stocks 44bp, crypto 10bp. These are not actual account fees/slippage. Funding omitted as in the baseline.",
            "Shared pool retains the original signal-timestamp slot clock, while crypto execution is next-bar open. Daily drawdown uses realized PNL, not intraday marked equity.",
            "Original triple-barrier conventions retained, including zero gross return when both barriers touch in one candle and checking the timeout bar before its opening-price timeout.",
            "Recent comparison starts flat after 386 warmup bars and excludes entries in the last 30h for identical complete-outcome coverage; it is a small portability study, not evidence of robust improvement.",
        ],
    )
    (args.output / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    report = ["# OKX 数据回测对照（研究，未改实盘）", "",
              f"原组合冻结区间：{START} 至 {END}。股票规则不变，600bp 触发。", "",
              "| 相同区间对照 | 组合累计收益 | 日度已实现最大回撤 | 组合交易数 | 加密单独收益（5槽） |",
              "|---|---:|---:|---:|---:|"]
    for label, r in (("原冻结组合", baseline), ("原加密信号 + OKX 成交价格", portability)):
        c = r["combined"]
        report.append(f"| {label} | {c['return_pct']:.4f}% | {c['max_dd_pct']:.4f}% | {c['trades']} | {r['crypto']['return_pct']:.4f}% |")
    report.extend(["", f"近期 OKX 输入研究：{recent_start} 至 {recent_end}；最后允许入场 {entry_end}。", "",
                   "| 近期相同窗口对照 | 组合累计收益 | 加密交易数 | 加密胜率 | 加密单独收益（5槽） |",
                   "|---|---:|---:|---:|---:|"])
    for label, r in (("屏蔽指标的对照组（非实盘精确复刻）", limited_result), ("补入可获取指标（仍缺成交笔数）", native_result)):
        c, q = r["combined"], r["crypto"]
        report.append(f"| {label} | {c['return_pct']:.4f}% | {q['trades']} | {q['win_pct']:.2f}% | {q['return_pct']:.4f}% |")
    report.extend(["", "## 数据与解释边界", "",
                   "8 币 K 线各 20,961 条（2月1日开始预热），无缺口。40 组衍生指标各 1,439 条，自 8月23日08:30 UTC 开始。",
                   "旧区间并非完整 OKX 原生信号回测；3月的15分钟衍生指标接口返回空，4小时数据不能等价替代。",
                   "近期沿用旧 Binance 训练模型与冻结阈值，没有重训或按结果选参；不同交易所的大户群体定义、量级可能不同。",
                   "成交笔数未获取，trade_size_z 保留缺失。详情、逐币缺失率、相关系数见 summary.json。", ""])
    report.extend(f"- {c}" for c in result["caveats"])
    (args.output / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
