"""Ablations of the frozen profitable combination, with baseline assertions.

Offline only. Retains the historical two-stage admission and realized-equity
accounting. It deliberately does not import any live/account client.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analysis.entry_rule_study import Policy, VARIANTS, filter_reason
from scripts.combinations.run import QUANT_SYMBOLS, _pooled_curve
from strategies.stocks.config import Config
from strategies.stocks.market import data
from strategies.stocks.research.backtest import Rules, run

START = pd.Timestamp("2026-03-03T09:30:00Z")
END = pd.Timestamp("2026-09-03T14:30:00Z")
DAYS = pd.date_range(START.normalize(), END.normalize(), freq="D")
STEP = pd.Timedelta(minutes=15)


def neutral_exit(frame, neutral, after, before, step):
    """Completed, contiguous neutral bars; execute on the next available open.

    frame is close-time indexed. No future close enters the decision. The
    baseline stop/deadline takes precedence at an equal execution timestamp.
    """
    count = 0
    previous = None
    for i in range(frame.index.searchsorted(after, side="right"), len(frame) - 1):
        stamp = frame.index[i]
        if stamp >= before:
            break
        if previous is not None and stamp - previous != step:
            count = 0
        count = count + 1 if bool(neutral[i]) else 0
        previous = stamp
        if count >= 3:
            execution = frame.index[i + 1] - step
            price = float(frame.open.iloc[i + 1])
            if stamp <= execution < before and np.isfinite(price) and price > 0:
                return execution, price
    return None


def stock_candidates():
    events_path = Path("results/stocks_offhours_research/events.csv")
    events = pd.read_csv(events_path)
    for c in ("event_ts", "resolve_ts"):
        events[c] = pd.to_datetime(events[c], utc=True)
    universe = pd.read_csv("data/stocks_swap/universe.csv")
    frames = data.to_bar_end(data.load_panel(sorted(universe.instId), "5m", "data/stocks_swap"), "5m")
    rules = Rules(horizon="to_open", resolve_offset_minutes=60, stop_loss_bps=300,
                  max_concurrent=100000, max_per_day=100000,
                  rank_column="abs_deviation", min_trailing_volume=0)
    candidates = run(events, frames, Config(), rules).trades
    # Unbounded capacity still enforces same-symbol nonoverlap in run(); use
    # independent per-event scans so no rejected shadow trade hides a candidate.
    if len(candidates) != len(events):
        pieces = [run(events.iloc[[i]], frames, Config(), rules).trades for i in range(len(events))]
        candidates = pd.concat(pieces, ignore_index=True)
    candidates = candidates.merge(events[["inst_id", "event_ts", "category", "deviation",
                                           "hours_to_open", "volume_ratio"]],
                                  left_on=["inst_id", "entry_ts"], right_on=["inst_id", "event_ts"],
                                  validate="one_to_one")
    candidates = candidates.assign(symbol=candidates.inst_id, asset="stock", strategy="xstock_hybrid",
                                   signal_strength=candidates.signal.abs())
    for field in ("exit_ts", "exit_price", "gross", "reason"):
        candidates[f"early_{field}"] = candidates[field]
    for i, row in candidates.iterrows():
        frame = frames[row.symbol]
        frame = frame.loc[(frame.index > row.entry_ts) &
                          (frame.index <= row.exit_ts + pd.Timedelta(minutes=5))]
        anchor = row.entry_price / (1 + row.deviation)
        neutral = (frame.close / anchor - 1).abs().le(.03).to_numpy()
        result = neutral_exit(frame, neutral, row.entry_ts, row.exit_ts, pd.Timedelta(minutes=5))
        if result:
            stamp, price = result
            candidates.loc[i, ["early_exit_ts", "early_exit_price", "early_gross", "early_reason"]] = [
                stamp, price, row.side * (price / row.entry_price - 1), "signal_invalid"]
    return candidates, frames, {"events_sha256": hashlib.sha256(events_path.read_bytes()).hexdigest(),
                                "stock_frames": len(frames), "stock_events": len(events)}


def crypto_candidates():
    path = Path("results/crypto/oos_dir_c_roll730.csv.gz")
    oos = pd.read_csv(path)
    oos["dt"] = pd.to_datetime(oos.dt, utc=True)
    rows, frames = [], {}
    for symbol in QUANT_SYMBOLS:
        group = oos.loc[oos.symbol == symbol].sort_values("ts").reset_index(drop=True)
        raw = pd.read_csv(f"data/klines/{symbol}.csv.gz", usecols=["ts", "open", "close"])
        raw.index = pd.to_datetime(raw.ts, unit="ms", utc=True)
        frames[symbol] = raw.set_axis(raw.index + STEP)
        neutral = (group.p_up.gt(group["lo_0.01"]) & group.p_up.lt(group["hi_0.01"])).to_numpy()
        eligible = group.loc[(group.p_up.ge(group["hi_0.01"]) | group.p_up.le(group["lo_0.01"])) &
                             np.isfinite(group.held_c) & np.isfinite(group.exit_ret_c)]
        for i, row in eligible.iterrows():
            side = 1 if row.p_up >= row["hi_0.01"] else -1
            stamp = row["dt"]
            base_exit = stamp + (row.held_c + 1) * STEP
            candidate = dict(strategy="okx_quant_c_tail_0.01", symbol=symbol, asset="crypto", category="crypto",
                             entry_ts=stamp, exit_ts=base_exit, side=side, gross=side * row.exit_ret_c,
                             entry_price=float(raw.open.get(stamp + STEP, np.nan)),
                             signal_strength=abs(row.p_up - .5), reason="original_label",
                             source_i=i, base_free_i=i + int(row.held_c) + 1)
            candidate.update(early_exit_ts=base_exit, early_gross=candidate["gross"],
                             early_reason=candidate["reason"], early_free_i=candidate["base_free_i"])
            # Earlier history is needed for exact per-symbol admission; no
            # position there can survive to the experiment window.
            if START - pd.Timedelta(days=2) <= stamp <= END:
                sub = group.iloc[i + 1:i + int(row.held_c) + 2]
                close_index = pd.DatetimeIndex(sub.dt + STEP)
                frame = pd.DataFrame({"open": raw.open.reindex(pd.DatetimeIndex(sub.dt)).to_numpy()},
                                     index=close_index)
                result = neutral_exit(frame, neutral[i + 1:i + 1 + len(sub)],
                                      stamp + STEP, base_exit, STEP)
                if result:
                    stamp, price = result
                    entry = float(raw.open.get(row["dt"] + STEP, np.nan))
                    if not np.isfinite(entry) or entry <= 0:
                        raise ValueError(f"Missing crypto entry price: {symbol} {row['dt']}")
                    candidate.update(early_exit_ts=stamp, early_gross=side * (price / entry - 1),
                                     early_reason="signal_invalid",
                                     early_free_i=int(group.dt.searchsorted(stamp)))
            rows.append(candidate)
    return pd.DataFrame(rows), frames, {"oos_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                       "oos_end": oos.dt.max().isoformat()}


def sleeve_select(candidates, policy):
    selected, active, per_day, free = [], [], {}, {}
    ordered = candidates.sort_values(["entry_ts", "signal_strength", "symbol"], ascending=[True, False, True])
    for row in ordered.to_dict("records"):
        if filter_reason(row, policy):
            continue
        now = row["entry_ts"]
        if row["asset"] == "crypto":
            if row["source_i"] < free.get(row["symbol"], 0):
                continue
            free[row["symbol"]] = row["early_free_i"] if policy.signal_exit else row["base_free_i"]
        else:
            active = [r for r in active if r["exit_ts"] > now]
            day = now.normalize()
            if len(active) >= 3 or per_day.get(day, 0) >= 2 or any(r["symbol"] == row["symbol"] for r in active):
                continue
            per_day[day] = per_day.get(day, 0) + 1
        if policy.signal_exit:
            row.update(exit_ts=row["early_exit_ts"], gross=row["early_gross"], reason=row["early_reason"])
            if row["asset"] == "stock":
                row["exit_price"] = row["early_exit_price"]
        row["hold_hours"] = (row["exit_ts"] - now).total_seconds() / 3600
        if row["asset"] == "stock":
            active.append(row)
        # Horizon-complete inclusion uses ORIGINAL exits for all variants.
        if now >= START and row["original_exit_ts"] <= END:
            selected.append(row)
    return pd.DataFrame(selected)


def pooled(trades, policy, days=DAYS):
    ordered = trades.sort_values(["entry_ts", "signal_strength", "strategy"], ascending=[True, False, True]).reset_index(drop=True)
    timeline = []
    for i, row in ordered.iterrows():
        timeline.extend([(row.entry_ts, 1, i), (row.exit_ts, 0, i)])
    timeline.sort(key=lambda x: (x[0], x[1]))
    equity, active, accepted, rejected, changes = 1., {}, [], [], []
    for stamp, kind, i in timeline:
        row = ordered.iloc[i]
        if not kind:
            if i in active:
                equity += active.pop(i) * row.net
                changes.append((stamp, equity))
            continue
        reason = None
        if len(active) >= policy.slots:
            reason = "shared_capacity"
        elif policy.concentration:
            if sum(ordered.iloc[j].side == row.side for j in active) >= policy.max_direction:
                reason = "direction_limit"
            elif sum(ordered.iloc[j].category == row.category for j in active) >= policy.max_category:
                reason = "category_limit"
        if reason:
            rejected.append(dict(entry_ts=stamp, symbol=row.symbol, reason=reason))
            continue
        active[i] = equity / policy.slots
        ordered.loc[i, "notional"] = active[i]
        accepted.append(i)
    if changes:
        stamps, values = zip(*changes)
        curve = pd.Series(values, index=stamps).resample("1D").last().reindex(days).ffill().fillna(1.)
    else:
        curve = pd.Series(1., index=days)
    return curve, ordered.iloc[accepted].copy(), pd.DataFrame(rejected)


def stats(curve, trades):
    return dict(trades=len(trades), return_pct=(curve.iloc[-1] - 1) * 100,
                max_dd_pct=(curve / curve.cummax() - 1).min() * 100,
                win_pct=(trades.net > 0).mean() * 100,
                avg_net_bps=trades.net.mean() * 1e4,
                signal_exits=int(trades.reason.eq("signal_invalid").sum()))


def marked_curve(trades, frames, cutoffs):
    """Mark open trades using completed bars and the same admitted notionals.

    Does not resize the original portfolio; its realized-equity sizing is
    intentionally unchanged. Half of the modeled cost is paid at each end.
    """
    values = np.ones(len(cutoffs))
    for row in trades.itertuples():
        entry = row.entry_ts + (STEP if row.asset == "crypto" else pd.Timedelta(0))
        cost = row.gross - row.net
        values[cutoffs >= row.exit_ts] += row.notional * row.net
        opened = (cutoffs >= entry) & (cutoffs < row.exit_ts)
        if opened.any():
            frame = frames[row.symbol]
            positions = frame.index.searchsorted(cutoffs[opened], side="right") - 1
            if (positions < 0).any() or not np.isfinite(row.entry_price) or row.entry_price <= 0:
                raise ValueError(f"Missing mark/entry for {row.symbol} {entry}")
            marks = frame.close.iloc[positions].to_numpy()
            values[opened] += row.notional * (row.side * (marks / row.entry_price - 1) - cost / 2)
    return pd.Series(values, index=cutoffs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/original_rule_study_20260907"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    stocks, stock_frames, meta = stock_candidates()
    print(f"Original stock candidates: {len(stocks)}", flush=True)
    crypto, crypto_frames, crypto_meta = crypto_candidates()
    print(f"Original crypto candidates: {len(crypto)}", flush=True)
    for frame in (stocks, crypto):
        frame["original_exit_ts"] = frame.exit_ts
    stocks.to_csv(args.output / "stock_candidates.csv", index=False)
    crypto.to_csv(args.output / "crypto_candidates.csv", index=False)
    frames = {**stock_frames, **crypto_frames}
    cutoffs = pd.DatetimeIndex([START - pd.Timedelta(seconds=1),
                                *pd.date_range(START.normalize() + pd.Timedelta(days=1), END.normalize(), freq="D"), END])
    summaries, segments, components = [], [], []
    for name, policy in VARIANTS.items():
        stock = sleeve_select(stocks, policy)
        quant = sleeve_select(crypto, policy)
        quant["net"] = quant.gross - .001
        _, quant, _ = _pooled_curve(quant, DAYS, 5)
        for cost_name, stock_bps, crypto_bps in [("original_cost", 44, 10), ("lower_fee_scenario", 34, 10), ("stress", 68, 16)]:
            combined = pd.concat([stock, quant], ignore_index=True)
            combined["net"] = combined.gross - np.where(combined.asset.eq("stock"), stock_bps, crypto_bps) / 1e4
            curve, trades, rejected = pooled(combined, policy)
            row = dict(variant=name, cost=cost_name, **stats(curve, trades))
            mtm = marked_curve(trades, frames, cutoffs)
            np.testing.assert_allclose(mtm.iloc[-1], curve.iloc[-1], atol=1e-10)
            row["daily_mtm_max_dd_pct"] = (mtm / mtm.cummax() - 1).min() * 100
            summaries.append(row)
            if name == "baseline" and cost_name == "original_cost":
                saved = pd.read_csv("results/combinations/latest/combined_trades.csv")
                for c in ("entry_ts", "exit_ts"):
                    saved[c] = pd.to_datetime(saved[c], utc=True)
                cols = ["entry_ts", "exit_ts", "symbol", "side", "gross", "net"]
                sort = ["entry_ts", "symbol"]
                pd.testing.assert_frame_equal(trades[cols].sort_values(sort).reset_index(drop=True),
                                              saved[cols].sort_values(sort).reset_index(drop=True), check_dtype=False)
                assert len(trades) == 299
                np.testing.assert_allclose(row["return_pct"], 50.27715375, atol=1e-7)
                np.testing.assert_allclose(row["max_dd_pct"], -8.35453768, atol=1e-7)
                print("PASS: baseline exactly reproduces all 299 original trades and +50.27715375%", flush=True)
            for label, lo, hi in [("before_august", START, pd.Timestamp("2026-08-01T00:00:00Z")),
                                   ("august_onward", pd.Timestamp("2026-08-01T00:00:00Z"), END + pd.Timedelta(seconds=1))]:
                selected = trades.loc[(trades.entry_ts >= lo) & (trades.entry_ts < hi)]
                # Rebase accepted trades by entry cohort; no claim of unseen OOS.
                subcurve, subtrades, _ = pooled(selected, Policy())
                segments.append(dict(variant=name, cost=cost_name, segment=label, **stats(subcurve, subtrades)))
            if cost_name == "original_cost":
                trades.to_csv(args.output / f"trades_{name}.csv", index=False)
                rejected.to_csv(args.output / f"rejected_{name}.csv", index=False)
                curve.to_csv(args.output / f"equity_{name}.csv", header=["equity"])
                mtm.to_csv(args.output / f"marked_equity_{name}.csv", header=["equity"])
                for asset, group in trades.groupby("asset"):
                    components.append(dict(variant=name, asset=asset, trades=len(group),
                                           pnl_pct=(group.notional * group.net).sum() * 100,
                                           win_pct=group.net.gt(0).mean() * 100))
                print(json.dumps(row), flush=True)
    summary = pd.DataFrame(summaries)
    summary.to_csv(args.output / "summary.csv", index=False)
    pd.DataFrame(segments).to_csv(args.output / "segments.csv", index=False)
    pd.DataFrame(components).to_csv(args.output / "components.csv", index=False)
    meta.update(crypto_meta, start=START.isoformat(), end=END.isoformat(), policies={k: asdict(v) for k, v in VARIANTS.items()},
                baseline_asserted=True, stock_trigger_bps=600, stock_stop_bps=300, stock_take_bps=0,
                stock_exit="min(cash_open+60m,entry+30h)", stock_slots=3, stock_entries_utc_day=2,
                shared_slots=5, entry_weight=.2, accounting="net PNL booked at exit; daily realized-equity drawdown",
                caveats=["Retrospective sample, not untouched out-of-sample validation.",
                         "Original two-stage shadow admission retained; not a unified live allocator.",
                         "Original crypto labels and signal-start entry timestamps retained (actual entry is next open).",
                         "Stock stop fills retain original exact barrier price, without gap-through penalty.",
                         "Funding, live liquidity and impact not modeled; lower-fee scenario is not measured total cost.",
                         "Common frozen coverage ends September 3, not September 7."])
    (args.output / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    table = summary.loc[summary.cost.eq("original_cost")].to_string(index=False, float_format=lambda x: f"{x:.3f}")
    (args.output / "report.md").write_text("# Original Combination Ablation\n\n```text\n" + table +
        "\n```\n\nSee metadata.json for unchanged rules and limitations; segments.csv for retrospective cohorts.\n", encoding="utf-8")


if __name__ == "__main__":
    main()
