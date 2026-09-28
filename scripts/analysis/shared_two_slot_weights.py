"""Compound fixed two-slot books at requested weights, with execution stress."""
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analysis.stock_portfolio_robustness import (
    CRYPTO_STEP, drawdown_durations, portfolio_curve,
)
from scripts.analysis.stock_robustness_study import PERIODS, STEP, load_frames, metrics

OUT = Path("results/shared_two_slot_weights_20260928")
WEIGHTS = (.20, .25, .30, .35, .40)
CAPACITY = Path("results/stock_capacity_confirmation_20260928")
STOCK = Path("results/stock_robustness_confirmation_20260928")
PROTOCOL = Path("docs/shared_two_slot_weights_protocol_20260928.md")
SHARPE_PROTOCOL = Path("docs/shared_two_slot_sharpe_september_protocol_20260928.md")
WINDOWS = {
    "historical_shared_unfunded": PERIODS["historical"],
    "july_august_shared_stock_funded": (pd.Timestamp("2026-07-01T00:00Z"), pd.Timestamp("2026-09-01T00:00Z")),
    "september_stock_only_funded": PERIODS["september"],
}


def daily_risk(curve, initial_equity=1.):
    """Full UTC calendar-day returns; no partial days or invented missing marks."""
    if (curve.empty or not isinstance(curve.index, pd.DatetimeIndex)
            or curve.index.tz is None or curve.index.has_duplicates
            or not curve.index.is_monotonic_increasing
            or not np.isfinite(curve).all() or curve.le(0).any()
            or not np.isfinite(initial_equity) or initial_equity <= 0):
        raise ValueError("Daily risk requires positive, ordered, timezone-aware equity")
    curve = curve.copy()
    curve.index = curve.index.tz_convert("UTC")
    boundaries = pd.date_range(curve.index[0].ceil("1D"), curve.index[-1].floor("1D"), freq="1D")
    marks = curve.reindex(boundaries)
    if marks.isna().any():
        raise ValueError("Missing midnight equity; do not forward-fill daily risk")
    if len(marks) and boundaries[0] == curve.index[0]:
        # Include costs of an entry exactly at the initial portfolio boundary.
        marks.iloc[0] = initial_equity
    returns = marks.pct_change(fill_method=None).iloc[1:].rename("daily_return")
    sd = returns.std(ddof=1)
    stats = dict(sharpe_daily=float(returns.mean()/sd*np.sqrt(365)) if sd > 0 else np.nan,
        sharpe_days=len(returns), daily_mean_pct=returns.mean()*100,
        daily_vol_pct=sd*100, annualized_vol_pct=sd*np.sqrt(365)*100,
        sharpe_start_utc=str(boundaries[0]) if len(returns) else None,
        sharpe_end_utc=str(boundaries[-1]) if len(returns) else None)
    return returns, stats


def load_book(path, stock_only=False):
    frame = pd.read_csv(path)
    for col in ("entry_ts", "exit_ts", "coverage_deadline"):
        frame[col] = pd.to_datetime(frame[col], utc=True)
    if stock_only:
        frame["asset"] = "stock"
    frame["funding_events"] = frame.funding_events.map(ast.literal_eval)
    if not frame.entry_ts.is_monotonic_increasing or frame.duplicated(["symbol", "entry_ts"]).any():
        raise ValueError("Frozen book is not an ordered, unique admission list")
    if not frame.asset.isin(["stock", "crypto"]).all():
        raise ValueError("Unknown asset class")
    return frame


def resize_book(book, weight, stock_cost):
    if not np.isfinite(weight) or not 0 < weight <= .5 or stock_cost not in (44, 68):
        raise ValueError("Expected positive two-slot weight <=50% and a declared cost pair")
    frame = book.copy()
    frame["cost_bps"] = np.where(frame.asset.eq("stock"), stock_cost, 16 if stock_cost == 68 else 10)
    frame["net"] = frame.gross+frame.funding-frame.cost_bps/10000
    frame["position_weight"] = weight
    equity, active = 1., []
    for index, row in frame.iterrows():
        done = sorted([r for r in active if r["exit_ts"] <= row.entry_ts], key=lambda r: r["exit_ts"])
        equity += sum(r["notional"]*r["net"] for r in done)
        active = [r for r in active if r["exit_ts"] > row.entry_ts]
        if equity <= 0 or not np.isfinite(equity):
            raise ValueError("Sizing equity exhausted; a solvent-return simulation cannot continue")
        if len(active) >= 2 or any(r["symbol"] == row.symbol for r in active):
            raise ValueError("Frozen book violates two-slot/nonoverlap admission")
        notional = equity*weight
        frame.loc[index, "notional"] = notional
        frame.loc[index, "entry_sizing_equity"] = equity
        active.append(dict(symbol=row.symbol, exit_ts=row.exit_ts, notional=notional, net=row.net))
    return frame


def capital_usage(trades, stocks, crypto, curve):
    clock = curve.index
    committed, current, count = np.zeros(len(clock)), np.zeros(len(clock)), np.zeros(len(clock))
    if curve.le(0).any():
        raise ValueError("Nonpositive marked equity; exposure ratios are undefined")
    for row in trades.itertuples():
        first, last = clock.searchsorted(row.entry_ts), clock.searchsorted(row.exit_ts)
        f = (stocks if row.asset == "stock" else crypto)[row.symbol]
        step = STEP if row.asset == "stock" else CRYPTO_STEP
        positions = (f.index+step).searchsorted(clock[first:last], side="right")-1
        if (positions < 0).any() or ((clock[first:last]-(f.index+step)[positions]) >= step).any():
            raise ValueError(f"Missing causal exposure marks: {row.symbol}")
        prices = f.close.to_numpy()[positions].copy()
        if len(prices) and clock[first] == row.entry_ts:
            prices[0] = row.entry_price
        committed[first:last] += row.notional
        current[first:last] += row.notional*prices/row.entry_price
        count[first:last] += 1
    detail = pd.DataFrame(dict(equity=curve, active_positions=count,
        entry_notional_ratio=committed/curve.to_numpy(),
        current_notional_ratio=current/curve.to_numpy()), index=clock)
    detail["one_x_headroom_proxy"] = 1-detail.current_notional_ratio
    return detail, dict(max_positions=int(count.max()),
        max_entry_notional_pct=detail.entry_notional_ratio.max()*100,
        max_current_notional_pct=detail.current_notional_ratio.max()*100,
        min_one_x_headroom_proxy_pct=detail.one_x_headroom_proxy.min()*100,
        negative_headroom_observations=int(detail.one_x_headroom_proxy.lt(0).sum()),
        average_entry_notional_pct=detail.entry_notional_ratio.mean()*100,
        time_in_market_pct=float((count > 0).mean()*100))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    stocks, _, _, hashes = load_frames()
    hashes[str(PROTOCOL)] = hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
    hashes[str(SHARPE_PROTOCOL)] = hashlib.sha256(SHARPE_PROTOCOL.read_bytes()).hexdigest()
    books = {}
    for scenario in WINDOWS:
        for delay in (0, 5, 10):
            if scenario == "september_stock_only_funded":
                path = STOCK / f"trades_stock_cap_2_{delay}_funded_conservative_september_68.csv"
            else:
                source = "historical_unfunded" if scenario.startswith("historical") else "july_august_stock_funded"
                path = CAPACITY / f"trades_{source}_{delay}_2_unified.csv"
            hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            books[scenario, delay] = load_book(path, stock_only=scenario == "september_stock_only_funded")
    historical = books["historical_shared_unfunded", 0]
    crypto = {}
    start, end = PERIODS["historical"]
    symbols = set().union(*(set(book.loc[book.asset.eq("crypto"), "symbol"]) for book in books.values()))
    for symbol in sorted(symbols):
        path = Path("data/klines") / f"{symbol}.csv.gz"
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        frame = pd.read_csv(path, usecols=["ts", "open", "close"])
        frame.index = pd.to_datetime(frame.ts, unit="ms", utc=True)
        frame = frame[(frame.index >= start-pd.Timedelta(days=1)) & (frame.index <= end)].sort_index()
        assert not frame.index.has_duplicates
        assert not len(pd.date_range(frame.index.min(), frame.index.max(), freq=CRYPTO_STEP).difference(frame.index))
        crypto[symbol] = frame
    summaries, months, components, controls = [], [], [], []
    for scenario, (a, b) in WINDOWS.items():
        for delay in (0, 5, 10):
            book = books[scenario, delay]
            for stock_cost in (44, 68):
                for weight in WEIGHTS:
                    trades = resize_book(book, weight, stock_cost)
                    assert trades[["symbol", "entry_ts", "exit_ts"]].equals(book[["symbol", "entry_ts", "exit_ts"]])
                    curve = portfolio_curve(trades, stocks, crypto, a, b)
                    assert np.isclose(curve.iloc[-1], 1+(trades.notional*trades.net).sum())
                    usage, usage_metrics = capital_usage(trades, stocks, crypto, curve)
                    daily = curve.resample("1D").last()
                    daily_returns = daily/daily.shift().fillna(1)-1
                    full_day_returns, risk = daily_risk(curve)
                    ends = curve[curve.index < b].groupby(lambda stamp: stamp.strftime("%Y-%m")).last()
                    changes = (ends/ends.shift().fillna(1)-1)*100
                    row = dict(scenario=scenario, delay_minutes=delay, stock_cost_bps=stock_cost,
                               weight_pct=int(round(weight*100)), target_two_slot_pct=weight*200,
                               **metrics(trades, curve), **drawdown_durations(curve), **usage_metrics, **risk,
                               worst_day_pct=daily_returns.min()*100, worst_month_pct=changes.min(),
                               worst_trade_entry_equity_pct=(weight*trades.net).min()*100)
                    summaries.append(row)
                    tag = f"{scenario}_{delay}_{stock_cost}_{int(round(weight*100))}"
                    trades.to_csv(OUT / f"trades_{tag}.csv", index=False)
                    daily.to_csv(OUT / f"daily_{tag}.csv")
                    full_day_returns.to_csv(OUT / f"daily_returns_{tag}.csv")
                    usage.to_csv(OUT / f"capital_{tag}.csv.gz", compression="gzip")
                    months.extend(dict(scenario=scenario, delay_minutes=delay, stock_cost_bps=stock_cost,
                                       weight_pct=row["weight_pct"], month=m, return_pct=r) for m, r in changes.items())
                    for asset, group in trades.groupby("asset"):
                        components.append(dict(scenario=scenario, delay_minutes=delay, stock_cost_bps=stock_cost,
                            weight_pct=row["weight_pct"], asset=asset, trades=len(group),
                            contribution_pct=(group.notional*group.net).sum()*100))
                    if weight == .2 and stock_cost == 68:
                        np.testing.assert_allclose(trades.notional, book.notional, atol=1e-10)
                        np.testing.assert_allclose(trades.net, book.net, atol=1e-12)
                        controls.append(dict(scenario=scenario, delay_minutes=delay,
                                             trades=len(trades), notional_net_reproduced=True))
            print(f"Finished {scenario}, delay={delay}m", flush=True)
    result = pd.DataFrame(summaries)
    control_path = Path("results/stock_portfolio_robustness_20260928/capacity_summary.csv")
    hashes[str(control_path)] = hashlib.sha256(control_path.read_bytes()).hexdigest()
    parent = pd.read_csv(control_path).query("scope == 'shared' and slots == 2 and sizing == 'fixed_20pct'")
    fresh = result.query("scenario == 'historical_shared_unfunded' and delay_minutes == 0 and weight_pct == 20")
    matched = parent.merge(fresh, left_on="cost", right_on="stock_cost_bps", suffixes=("_old", "_new"), validate="one_to_one")
    assert len(matched) == 2
    np.testing.assert_allclose(matched.return_pct_old, matched.return_pct_new, atol=1e-9)
    np.testing.assert_allclose(matched.marked_dd_pct_old, matched.marked_dd_pct_new, atol=1e-9)
    for _, group in result.groupby(["scenario", "delay_minutes", "stock_cost_bps"]):
        assert group.trades.nunique() == group.wins.nunique() == 1
        assert group.max_positions.le(2).all()
    assert len(result) == 90 and len(controls) == 9
    result.to_csv(OUT / "summary.csv", index=False)
    pd.DataFrame(months).to_csv(OUT / "monthly.csv", index=False)
    pd.DataFrame(components).to_csv(OUT / "components.csv", index=False)
    pd.DataFrame(controls).to_csv(OUT / "controls.csv", index=False)
    for source in (Path(__file__), Path("scripts/analysis/stock_portfolio_robustness.py"),
                   Path("scripts/analysis/stock_robustness_study.py")):
        hashes[str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
    (OUT / "manifest.json").write_text(json.dumps(dict(input_and_code_hashes=hashes,
        weights=WEIGHTS, rows=len(result), controls=len(controls), shared_slots=2,
        method="Fixed admitted books; each entry uses weight times then-current realized equity, not a linear return multiplier.",
        sharpe="UTC midnight-to-midnight marked returns, complete days only, idle days included; mean/std(ddof=1)*sqrt(365), risk-free=0. Undefined for fewer than two returns or zero variance.",
        caveats=["No live configuration or account action. All samples were previously inspected.",
                 "Admission is weight-independent only under the parent's simplified fractional-notional/no-margin-rejection model.",
                 "Fees/funding are realized on exit for sizing; marks and stock funding use the parent's causal equity model.",
                 "1x headroom is marked-equity minus proxy current gross notional, not OKX account available margin.",
                 "No historical mark-price margin, order loss, maintenance tiers, liquidation, lot rounding or size-dependent impact.",
                 "Historical window excludes funding; July-August includes stock funding only; September is stock-only.",
                 "Crypto predictions and outcomes are frozen Binance history, not the currently missing-feature live model.",
                 "Only stock entries are delayed; the fixed 5/10m scenarios are not measurements of actual live delay."]), indent=2))
    print(result.query("scenario == 'historical_shared_unfunded' and delay_minutes == 0").to_string(index=False))


if __name__ == "__main__":
    main()
