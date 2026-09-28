"""Frozen crypto comparator and matched two-stage/unified capacity diagnostics."""
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analysis.stock_funding_robustness import OUT as FUNDED, load_candidates
from scripts.analysis.stock_robustness_study import (
    OUT as STUDY, PERIODS, STEP, VARIANTS, entry_filter, load_frames, marked_curve, metrics,
)
from scripts.live.combination_policy import stock_underlying

OUT = Path("results/stock_portfolio_robustness_20260928")
CRYPTO_STEP = pd.Timedelta(minutes=15)
RULES = ("baseline", "displacement_le_8pct", "stock_cap_1", "stock_cap_2",
         "hynix_dedup", "direction_cap_2")


def prepare_crypto(raw):
    frame = raw.copy()
    for col in ("entry_ts", "exit_ts"):
        frame[col] = pd.to_datetime(frame[col], utc=True)
    frame["signal_ts"] = frame.entry_ts
    frame["entry_ts"] += CRYPTO_STEP
    # The frozen label scanner includes its final horizon bar; a barrier there
    # can be timestamped at signal+50 bars. Preserve it, and purge that horizon.
    frame["coverage_deadline"] = frame.signal_ts+50*CRYPTO_STEP
    if frame.exit_ts.gt(frame.coverage_deadline).any() or frame.exit_ts.lt(frame.entry_ts).any():
        raise ValueError("Crypto label exceeds the declared frozen-label horizon")
    frame["hold_h"] = (frame.exit_ts-frame.entry_ts).dt.total_seconds()/3600
    frame["funding"] = 0.
    frame["funding_events"] = [[] for _ in range(len(frame))]
    return frame


def admit(candidates, variant, cost, start, end, slots=5, enforce_sleeves=True, weight=.2):
    part = candidates[candidates.entry_ts.ge(start) & candidates.coverage_deadline.lt(end)]
    ordered = part.sort_values(["entry_ts", "signal_strength", "symbol"], ascending=[True, False, True])
    active, accepted, rejected, per_day, free_crypto = [], [], [], {}, {}
    realized = 1.
    for row in ordered.to_dict("records"):
        now = row["entry_ts"]
        for x in sorted([x for x in active if x["exit_ts"] <= now], key=lambda t: t["exit_ts"]):
            realized += x["notional"]*x["net"]
        active = [x for x in active if x["exit_ts"] > now]
        stocks = [x for x in active if x["asset"] == "stock"]
        is_stock = row["asset"] == "stock"
        reason = entry_filter(row, variant) if is_stock and enforce_sleeves else None
        if not reason and len(active) >= slots:
            reason = "shared_capacity"
        if not reason and any(x["symbol"] == row["symbol"] for x in active):
            reason = "same_instrument"
        if enforce_sleeves:
            if not reason and not is_stock and row["source_i"] < free_crypto.get(row["symbol"], 0):
                reason = "crypto_source_cooldown"
            if is_stock:
                if not reason and len(stocks) >= variant.stock_slots:
                    reason = "stock_capacity"
                if not reason and per_day.get(now.normalize(), 0) >= 2:
                    reason = "stock_daily_limit"
                if not reason and variant.underlying_limit and any(stock_underlying(x["symbol"]) == stock_underlying(row["symbol"]) for x in stocks):
                    reason = "same_underlying"
                if not reason and sum(x["side"] == row["side"] for x in stocks) >= variant.direction_cap:
                    reason = "stock_direction_capacity"
        if reason:
            rejected.append(dict(symbol=row["symbol"], entry_ts=now, reason=reason))
            continue
        fee = cost if is_stock else (16 if cost == 68 else 10)
        row.update(cost_bps=fee, net=row["gross"]+row.get("funding", 0.)-fee/10000,
                   notional=realized*weight, position_weight=weight)
        active.append(row)
        accepted.append(row)
        # Rejected shadow trades never consume a quota or crypto cooldown.
        if is_stock:
            per_day[now.normalize()] = per_day.get(now.normalize(), 0)+1
        else:
            free_crypto[row["symbol"]] = row["base_free_i"]
    cols = list(dict.fromkeys([*candidates.columns, "cost_bps", "net", "notional", "position_weight"]))
    return pd.DataFrame(accepted, columns=cols), pd.DataFrame(rejected)


def portfolio_curve(trades, stocks, crypto, start, end):
    curve = marked_curve(trades[trades.asset.eq("stock")], stocks, start, end)
    clock, values = curve.index, curve.to_numpy().copy()
    for r in trades[trades.asset.eq("crypto")].itertuples():
        first, last = clock.searchsorted(r.entry_ts), clock.searchsorted(r.exit_ts)
        values[last:] += r.notional*r.net
        f = crypto[r.symbol]
        indices = (f.index+CRYPTO_STEP).searchsorted(clock[first:last], side="right")-1
        if (indices < 0).any() or ((clock[first:last]-(f.index+CRYPTO_STEP)[indices]) >= CRYPTO_STEP).any():
            raise ValueError(f"Missing/stale crypto mark: {r.symbol}")
        prices = f.close.to_numpy()[indices].copy()
        if len(prices) and clock[first] == r.entry_ts:
            prices[0] = r.entry_price
        values[first:last] += r.notional*(r.side*(prices/r.entry_price-1)-r.cost_bps/20000)
    return pd.Series(values, index=clock, name="equity")


def occupancy(trades, start, end):
    clock = pd.date_range(start, end, freq=STEP)
    total, stock, gross_weight = np.zeros(len(clock)), np.zeros(len(clock)), np.zeros(len(clock))
    for row in trades.itertuples():
        a, b = clock.searchsorted(row.entry_ts), clock.searchsorted(row.exit_ts)
        total[a:b] += 1
        stock[a:b] += int(row.asset == "stock")
        # Relative to each trade's entry sizing book, not current marked equity.
        gross_weight[a:b] += row.position_weight
    return dict(max_positions=int(total.max()), max_stock_positions=int(stock.max()),
                time_in_market_pct=float((total > 0).mean()*100),
                average_sizing_book_exposure_pct=float(gross_weight.mean()*100))


def drawdown_durations(curve):
    peak, peak_time, start, trough_time, depth = 1., curve.index[0], None, None, 0.
    episodes = []
    for stamp, value in curve.items():
        if value >= peak:
            if start is not None:
                episodes.append((start, stamp, trough_time, depth, True))
            peak, peak_time, start, depth = value, stamp, None, 0.
        else:
            if start is None:
                start, trough_time = peak_time, stamp
            dd = value/peak-1
            if dd < depth:
                depth, trough_time = dd, stamp
    if start is not None:
        episodes.append((start, curve.index[-1], trough_time, depth, False))
    if not episodes:
        return dict(longest_drawdown_days=0., worst_drawdown_recovery_days=0., worst_drawdown_recovered=True)
    worst = min(episodes, key=lambda row: row[3])
    return dict(longest_drawdown_days=max((r[1]-r[0]).total_seconds()/86400 for r in episodes),
                worst_drawdown_recovery_days=(worst[1]-worst[2]).total_seconds()/86400 if worst[4] else np.nan,
                worst_drawdown_recovered=worst[4])


def capacity_sweep(stock, crypto, frames, crypto_frames, start, end):
    rows = []
    for scope, candidates in (("stock_only", stock), ("crypto_only", crypto),
                              ("shared", pd.concat([stock, crypto], ignore_index=True))):
        for slots in range(1, 6):
            variant = replace(VARIANTS[0], stock_slots=slots, direction_cap=slots) if scope == "stock_only" else VARIANTS[0]
            for sizing, weight in (("fixed_20pct", .2), ("equal_slot_budget", 1/slots)):
                for cost in (44, 68):
                    tr, rejected = admit(candidates, variant, cost, start, end, slots=slots, weight=weight)
                    curve = portfolio_curve(tr, frames, crypto_frames, start, end)
                    rows.append(dict(scope=scope, slots=slots, sizing=sizing, cost=cost,
                                     **metrics(tr, curve), **occupancy(tr, start, end), **drawdown_durations(curve)))
                    tag = f"capacity_{scope}_{slots}_{sizing}_{cost}"
                    tr.to_csv(OUT / f"trades_{tag}.csv", index=False)
                    rejected.to_csv(OUT / f"rejected_{tag}.csv", index=False)
                    curve.resample("1D").last().to_csv(OUT / f"daily_{tag}.csv")
    pd.DataFrame(rows).to_csv(OUT / "capacity_summary.csv", index=False)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    stocks, _, _, hashes = load_frames()
    path = Path("results/original_rule_study_20260907/crypto_candidates.csv")
    hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    crypto = prepare_crypto(pd.read_csv(path))
    start, end = PERIODS["historical"]
    crypto = crypto[crypto.entry_ts.ge(start) & crypto.entry_ts.lt(end)].copy()
    crypto_frames = {}
    for symbol in crypto.symbol.unique():
        path = Path("data/klines") / f"{symbol}.csv.gz"
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        frame = pd.read_csv(path, usecols=["ts", "open", "close"])
        frame.index = pd.to_datetime(frame.ts, unit="ms", utc=True)
        frame = frame[(frame.index >= start-pd.Timedelta(days=1)) & (frame.index <= end)].sort_index()
        expected = pd.date_range(frame.index.min(), frame.index.max(), freq=CRYPTO_STEP)
        if frame.index.has_duplicates or len(expected.difference(frame.index)):
            raise ValueError(f"Crypto candle gap: {symbol}")
        sub = crypto[crypto.symbol.eq(symbol)]
        np.testing.assert_allclose(frame.open.reindex(pd.DatetimeIndex(sub.entry_ts)), sub.entry_price, rtol=1e-10)
        crypto_frames[symbol] = frame
    # No optimization: these are the predeclared capacity/risk controls and the
    # entry rule flagged by the parent study. All cost/scenario outcomes saved.
    rows, components = [], []
    windows = {"historical": (start, end), "july_august": (pd.Timestamp("2026-07-01T00:00Z"), pd.Timestamp("2026-09-01T00:00Z"))}
    for scenario in ("unfunded", "stock_funded_crypto_unfunded"):
        periods = windows if scenario == "unfunded" else {"july_august": windows["july_august"]}
        for rule in RULES:
            variant = next(v for v in VARIANTS if v.name == rule)
            path = (STUDY if scenario == "unfunded" else FUNDED) / f"candidates_{rule}.csv"
            hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            stock = load_candidates(path).assign(asset="stock")
            if scenario == "unfunded":
                stock["funding"] = 0.
                stock["funding_events"] = [[] for _ in range(len(stock))]
            else:
                stock["funding_events"] = stock.funding_events.map(json.loads)
            all_candidates = pd.concat([stock, crypto], ignore_index=True)
            for period, (a, b) in periods.items():
                for cost in (44, 68):
                    scopes = [("stock_only", stock), ("unified", all_candidates)]
                    if rule == "baseline":
                        scopes.append(("crypto_only", crypto))
                    stock_sleeve, _ = admit(stock, variant, cost, a, b)
                    crypto_sleeve, _ = admit(crypto, variant, cost, a, b)
                    scopes.append(("two_stage", pd.concat([stock_sleeve, crypto_sleeve], ignore_index=True)))
                    for scope, candidates in scopes:
                        tr, rejected = admit(candidates, variant, cost, a, b, enforce_sleeves=scope != "two_stage")
                        curve = portfolio_curve(tr, stocks, crypto_frames, a, b)
                        assert np.isclose(curve.iloc[-1], 1+(tr.notional*tr.net).sum())
                        rows.append(dict(variant=rule, scenario=scenario, period=period, cost=cost, scope=scope,
                                         **metrics(tr, curve), **occupancy(tr, a, b)))
                        for asset, g in tr.groupby("asset"):
                            components.append(dict(variant=rule, scenario=scenario, period=period, cost=cost,
                                scope=scope, asset=asset, trades=len(g), contribution_pct=(g.notional*g.net).sum()*100))
                        tag = f"{rule}_{scenario}_{period}_{cost}_{scope}"
                        tr.to_csv(OUT / f"trades_{tag}.csv", index=False)
                        rejected.to_csv(OUT / f"rejected_{tag}.csv", index=False)
                        curve.resample("1D").last().to_csv(OUT / f"daily_{tag}.csv")
                        if scope == "stock_only" and scenario == "unfunded" and period == "historical":
                            parent = pd.read_csv(STUDY / "summary.csv")
                            expected = parent[(parent.variant == rule) & (parent.period == period) & (parent.cost == cost)].iloc[0]
                            assert len(tr) == expected.trades
                            assert np.isclose((curve.iloc[-1]-1)*100, expected.return_pct)
        print(f"Portfolio scenario finished: {scenario}", flush=True)
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "summary.csv", index=False)
    pd.DataFrame(components).to_csv(OUT / "components.csv", index=False)
    stock = load_candidates(STUDY / "candidates_baseline.csv").assign(asset="stock", funding=0.)
    stock["funding_events"] = [[] for _ in range(len(stock))]
    capacity_sweep(stock, crypto, stocks, crypto_frames, start, end)
    (OUT / "manifest.json").write_text(json.dumps(dict(input_hashes=hashes, variants=RULES,
        crypto_entry="Archive signal-bar start +15m; archived entry price checked against next open.",
        crypto_horizon="Common signal+50 bars purge preserves the frozen label scanner's last-bar convention; no label retraining.",
        allocation="Five shared slots, fixed 20% of realized equity per accepted trade; stock cap varies, no weight renormalization.",
        capacity_sweep="Separate 1-5 slots, fixed 20% versus equal 1/slots weights; stock-only cap=slots, shared stock cap=3.",
        caveats=["Frozen crypto predictions/outcomes from Binance history, not OKX executions or current live features.",
                 "Crypto label barriers/gaps and the extra final-bar convention are unchanged, so combined returns are not full execution-adjusted forecasts.",
                 "Crypto funding omitted. Funded scenario includes only stock funding; never call it fully funded portfolio net performance.",
                 "Two-stage and unified diagnostics here share causal clocks/common horizons; neither is the old reported combination.",
                 "Unified admission charges quotas/cooldown only for accepted trades; two-stage preselection can consume rejected shadow opportunities.",
                 "Fixed 20% realized-equity sizing is a research convention, not a margin/lot-size simulator.",
                 "Marked drawdown is on a 5m grid with 15m completed marks for crypto.",
                 "Retrospective diagnostic; no model, universe, live config or deployment changed."]), indent=2))
    print(result.query("cost == 68 and scenario == 'unfunded' and period == 'historical'").to_string(index=False))


if __name__ == "__main__":
    main()
