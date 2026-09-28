"""Predeclared causal stock execution/quality/exit study, with frozen folds."""
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analysis.strategy_replacement_study import path_exit
from scripts.live.combination_policy import stock_underlying
from strategies.stocks.config import Config
from strategies.stocks.market import reference

OUT = Path("results/stock_robustness_20260928")
STEP = pd.Timedelta(minutes=5)
START = pd.Timestamp("2026-03-03T09:30Z")
END = pd.Timestamp("2026-09-26T00:00Z")
PERIODS = {
    "historical": (START, pd.Timestamp("2026-09-03T14:30Z")),
    "development": (START, pd.Timestamp("2026-06-01T00:00Z")),
    "june_july": (pd.Timestamp("2026-06-01T00:00Z"), pd.Timestamp("2026-08-01T00:00Z")),
    "august": (pd.Timestamp("2026-08-01T00:00Z"), pd.Timestamp("2026-09-01T00:00Z")),
    "september": (pd.Timestamp("2026-09-01T00:00Z"), END),
    "recent": (pd.Timestamp("2026-09-11T00:00Z"), END),
}


@dataclass(frozen=True)
class Variant:
    name: str
    selectable: bool = True
    volume_24h: float = 0.
    signal_volume: float = 0.
    listing_days: float = 0.
    max_wait_h: float = float("inf")
    max_displacement: float = float("inf")
    entry_guard: bool = False
    resolve_minutes: int = 60
    max_hold_h: float = 30.
    target: str = "none"
    stop: float = .03
    overnight_only: bool = False
    stock_slots: int = 3
    underlying_limit: bool = False
    direction_cap: int = 3
    side: int = 0
    delay_bars: int = 0


VARIANTS = [
    Variant("baseline", selectable=False),
    Variant("liquidity", volume_24h=150000, signal_volume=1400),
    Variant("liquidity_strict", volume_24h=500000, signal_volume=5000),
    Variant("listing_7d", listing_days=7),
    Variant("open_within_12h", max_wait_h=12),
    Variant("displacement_le_8pct", max_displacement=.08),
    Variant("fresh_price_guard", entry_guard=True),
    Variant("exit_at_open", resolve_minutes=0),
    Variant("exit_open_plus_15m", resolve_minutes=15),
    Variant("hold_cap_12h", max_hold_h=12),
    Variant("half_anchor_target", target="half_anchor"),
    Variant("take_profit_3pct", target="fixed_3pct"),
    Variant("stop_2pct", stop=.02, selectable=False),
    Variant("overnight_only", overnight_only=True),
    Variant("stock_cap_1", stock_slots=1, selectable=False),
    Variant("stock_cap_2", stock_slots=2, selectable=False),
    Variant("hynix_dedup", underlying_limit=True, selectable=False),
    Variant("direction_cap_2", direction_cap=2, selectable=False),
    Variant("long_only", side=1, selectable=False),
    Variant("short_only", side=-1, selectable=False),
    Variant("liquidity_open_15m", volume_24h=150000, signal_volume=1400, resolve_minutes=15),
    Variant("liquidity_half_anchor", volume_24h=150000, signal_volume=1400, target="half_anchor"),
    Variant("entry_quality", max_wait_h=12, max_displacement=.08, entry_guard=True),
    Variant("delay_5m", delay_bars=1, selectable=False),
    Variant("delay_10m", delay_bars=2, selectable=False),
    Variant("delay_5m_guard", delay_bars=1, entry_guard=True, selectable=False),
]


def load_frames():
    universe = pd.read_csv("data/stocks_swap/universe.csv")
    frames, audit, hashes = {}, [], {}
    for row in universe.itertuples():
        inst = row.instId
        if reference.reference_market(inst) is None:
            continue
        paths = [Path("data/research_stock_history_20260928/5m") / f"{inst}.csv",
                 Path("results/live_audit_20260911/market/5m") / f"{inst}.csv",
                 Path("results/us_only_study_20260928") / f"recent_market_{inst}.csv.gz"]
        pieces, overlap = [], 0
        for i, path in enumerate(paths):
            f = pd.read_csv(path)
            f["ts"] = pd.to_datetime(f.ts, utc=True)
            if i == 2:
                f["ts"] -= STEP  # This snapshot was already bar-close indexed.
            f = f.set_index("ts").sort_index()
            f.index = f.index.as_unit("ns")
            if pieces:
                previous = pd.concat(pieces).loc[lambda d: ~d.index.duplicated()]
                common = previous.index.intersection(f.index)
                cols = ["open", "high", "low", "close", "volume_quote"]
                if not np.allclose(previous.loc[common, cols], f.loc[common, cols], rtol=1e-9, atol=1e-9):
                    raise ValueError(f"Conflicting snapshots: {inst}/{path}")
                overlap += len(common)
            pieces.append(f)
            hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        f = pd.concat(pieces).loc[lambda d: ~d.index.duplicated()].sort_index()
        f = f[f.index < END]
        if f.index.min() < pd.Timestamp(row.list_ts).floor("5min"):
            raise ValueError(f"Pre-listing data: {inst}")
        expected = pd.date_range(f.index.min(), END-STEP, freq=STEP)
        audit.append(dict(inst=inst, first=str(f.index.min()), last=str(f.index.max()), rows=len(f),
                          overlapping_rows=overlap, missing_bars=len(expected.difference(f.index))))
        frames[inst] = f
    return frames, universe, pd.DataFrame(audit), hashes


def add_event_features(events, frames, universe):
    out = events.copy()
    specs = universe.set_index("instId")
    volumes = {k: pd.Series(f.volume_quote.to_numpy(), index=f.index+STEP).rolling("24h", min_periods=288).sum()
               for k, f in frames.items()}
    rows = []
    for e in out.itertuples():
        f = frames[e.inst_id]
        signal_bar = f.loc[e.event_ts-STEP]
        price = float(signal_bar.close)
        rows.append(dict(signal_price=price, anchor_price=price/(1+float(e.deviation)),
                         signal_volume=float(signal_bar.volume_quote),
                         volume_24h=float(volumes[e.inst_id].loc[e.event_ts]),
                         listing_days=(e.event_ts-pd.Timestamp(specs.loc[e.inst_id, "list_ts"])).total_seconds()/86400,
                         coverage_deadline=min(e.resolve_ts+pd.Timedelta(hours=1), e.event_ts+pd.Timedelta(hours=30))))
    return pd.concat([out.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def price_candidates(events, frames, variant, cache):
    records, excluded = [], []
    for event in events.to_dict("records"):
        entry = event["event_ts"]+variant.delay_bars*STEP
        deadline = min(event["resolve_ts"]+pd.Timedelta(minutes=variant.resolve_minutes),
                       event["event_ts"]+pd.Timedelta(hours=variant.max_hold_h))
        inst = event["inst_id"]
        f = frames[inst]
        reason = None
        if entry >= event["resolve_ts"] or deadline <= entry:
            reason = "expired"
        elif entry not in f.index or deadline not in f.index:
            reason = "missing_entry_or_exit"
        if reason:
            excluded.append(dict(inst=inst, event_ts=event["event_ts"], reason=reason))
            continue
        key = (inst, entry, deadline, variant.stop, variant.target)
        if key not in cache:
            px = float(f.loc[entry, "open"])
            take = 0.
            if variant.target == "fixed_3pct":
                take = .03
            elif variant.target == "half_anchor":
                target = (event["signal_price"]+event["anchor_price"])/2
                take = int(event["side"])*(target/px-1)
                if take <= 0:
                    cache[key] = None
            if variant.target != "half_anchor" or take > 0:
                # The shared scanner enters at bar open and fills adverse gaps.
                cache[key] = path_exit(f, entry, int(event["side"]), variant.stop, take, deadline, step=STEP)
        result = cache[key]
        if result is None:
            excluded.append(dict(inst=inst, event_ts=event["event_ts"], reason="gap_or_target_already_reached"))
            continue
        ep, xp, exit_ts, why = result
        # Price history establishing the signal anchor must also be contiguous.
        before = pd.date_range(event["anchor_close_ts"]-STEP, event["event_ts"]-STEP, freq=STEP)
        if len(before.difference(f.index)):
            excluded.append(dict(inst=inst, event_ts=event["event_ts"], reason="anchor_path_gap"))
            continue
        side = int(event["side"])
        records.append(dict(event, symbol=inst, entry_ts=entry, exit_ts=exit_ts, deadline=deadline,
                            exit_lower_ts=exit_ts-STEP if why in {"stop", "target"} else exit_ts,
                            entry_price=ep, exit_price=xp, gross=side*(xp/ep-1), reason=why,
                            signal_strength=abs(event["deviation"]),
                            hold_h=(exit_ts-entry).total_seconds()/3600,
                            remaining_bps=-side*(ep/event["anchor_price"]-1)*10000,
                            adverse_bps=side*(ep/event["signal_price"]-1)*10000))
    columns = list(events.columns)+["symbol", "entry_ts", "exit_ts", "deadline", "exit_lower_ts",
                                    "entry_price", "exit_price", "gross", "reason", "signal_strength",
                                    "hold_h", "remaining_bps", "adverse_bps"]
    return pd.DataFrame(records, columns=list(dict.fromkeys(columns))), pd.DataFrame(excluded)


def entry_filter(row, variant):
    if variant.volume_24h and (not np.isfinite(row["volume_24h"]) or row["volume_24h"] < variant.volume_24h):
        return "volume_24h"
    if row["signal_volume"] < variant.signal_volume:
        return "signal_volume"
    if row["listing_days"] < variant.listing_days:
        return "listing_age"
    if row["hours_to_open"] > variant.max_wait_h:
        return "wait_to_open"
    if abs(row["deviation"]) > variant.max_displacement:
        return "displacement"
    if variant.entry_guard and (row["remaining_bps"] < 600 or row["adverse_bps"] > 30):
        return "fresh_price"
    if variant.overnight_only and row["window_hours"] > 24:
        return "long_closure"
    if variant.side and row["side"] != variant.side:
        return "direction"
    return None


def replay(candidates, variant, cost, start, end):
    # Purge by the preknown common horizon, not a favorable realized early stop.
    part = candidates[candidates.entry_ts.ge(start) & candidates.coverage_deadline.lt(end)]
    ordered = part.sort_values(["entry_ts", "signal_strength", "symbol"], ascending=[True, False, True])
    active, accepted, rejected, per_day = [], [], [], {}
    equity = 1.
    for row in ordered.to_dict("records"):
        now = row["entry_ts"]
        closed = [x for x in active if x["exit_ts"] <= now]
        for x in sorted(closed, key=lambda t: t["exit_ts"]):
            equity += x["notional"]*x["net"]
        active = [x for x in active if x["exit_ts"] > now]
        reason = entry_filter(row, variant)
        if not reason and len(active) >= variant.stock_slots:
            reason = "stock_capacity"
        if not reason and any(x["symbol"] == row["symbol"] for x in active):
            reason = "same_instrument"
        if not reason and variant.underlying_limit and any(stock_underlying(x["symbol"]) == stock_underlying(row["symbol"]) for x in active):
            reason = "same_underlying"
        if not reason and sum(x["side"] == row["side"] for x in active) >= variant.direction_cap:
            reason = "direction_capacity"
        if not reason and per_day.get(now.normalize(), 0) >= 2:
            reason = "daily_limit"
        if reason:
            rejected.append(dict(symbol=row["symbol"], entry_ts=now, reason=reason))
            continue
        row.update(cost_bps=cost, net=row["gross"]+row.get("funding", 0.)-cost/10000, notional=equity*.2)
        accepted.append(row)
        active.append(row)
        per_day[now.normalize()] = per_day.get(now.normalize(), 0)+1
    return pd.DataFrame(accepted, columns=list(candidates.columns)+["cost_bps", "net", "notional"]), pd.DataFrame(rejected)


def marked_curve(trades, frames, start, end):
    clock = pd.date_range(start.ceil("5min"), end.floor("5min"), freq=STEP)
    value = np.ones(len(clock))
    for r in trades.itertuples():
        first = clock.searchsorted(r.entry_ts, side="left")
        last = clock.searchsorted(r.exit_ts, side="left")
        value[last:] += r.notional*r.net
        if last <= first:
            continue
        f = frames[r.symbol]
        indices = (f.index+STEP).searchsorted(clock[first:last], side="right")-1
        if (indices < 0).any():
            raise ValueError(f"Missing mark: {r.symbol}")
        stamps = (f.index+STEP)[indices]
        if ((clock[first:last]-stamps) > STEP).any():
            raise ValueError(f"Stale mark: {r.symbol}")
        prices = f.close.to_numpy()[indices].copy()
        if clock[first] == r.entry_ts:
            prices[0] = r.entry_price
        value[first:last] += r.notional*(r.side*(prices/r.entry_price-1)-r.cost_bps/20000)
        for stamp, cash in getattr(r, "funding_events", []):
            at = max(first, clock.searchsorted(pd.Timestamp(stamp), side="left"))
            if at < last:
                value[at:last] += r.notional*cash
    return pd.Series(value, index=clock, name="equity")


def metrics(trades, curve):
    net = trades.net.astype(float)
    contrib = (trades.notional*net).groupby(trades.symbol).sum()
    positive = contrib[contrib > 0]
    dd = curve/curve.cummax()-1
    return dict(trades=len(trades), wins=int(net.gt(0).sum()),
        win_pct=float(net.gt(0).mean()*100), return_pct=float((curve.iloc[-1]-1)*100),
        marked_dd_pct=float(dd.min()*100), mean_net_bps=float(net.mean()*10000),
        mean_hold_h=float(trades.hold_h.mean()), entry_days=int(trades.entry_ts.dt.normalize().nunique()) if len(trades) else 0,
        profit_factor=float(net[net>0].sum()/-net[net<0].sum()) if net.lt(0).any() else None,
        positive_symbols=len(positive), top_positive_symbol_share=float(positive.max()/positive.sum()) if len(positive) else None)


def choose_rule(training, baseline):
    eligible = training[(training.trades >= 20) & (training.entry_days >= 10) &
                        training.mean_net_bps.gt(0) &
                        training.marked_dd_pct.ge(-max(2., abs(baseline.marked_dd_pct)*1.25))]
    eligible = eligible[eligible.variant.isin([v.name for v in VARIANTS if v.selectable]+["baseline"])]
    if eligible.empty:
        return "cash"
    score = eligible.return_pct+eligible.marked_dd_pct
    best = score.idxmax()
    # Cash has zero return and drawdown, and remains a real alternative.
    return str(eligible.loc[best, "variant"]) if score.loc[best] > 0 else "cash"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    definitions = [asdict(v) for v in VARIANTS]
    (OUT / "variants.json").write_text(json.dumps(definitions, indent=2))
    frames, universe, audit, hashes = load_frames()
    audit.to_csv(OUT / "coverage.csv", index=False)
    print(f"Merged {len(frames)} contracts, {int(audit.rows.sum())} bars, {int(audit.missing_bars.sum())} internal/tail gaps", flush=True)
    close_frames = {k: f.set_axis(f.index+STEP) for k, f in frames.items()}
    events = reference.stock_events(close_frames, Config(dislocation_bps=600))
    events = events[events.event_ts.ge(START) & events.event_ts.lt(END)].copy()
    events = add_event_features(events, frames, universe)
    events.to_csv(OUT / "events.csv", index=False)
    print(f"Corrected-calendar events: {len(events)}", flush=True)
    from scripts.analysis.reference_calendar_full_history import independent_candidates
    old_events = events[events.event_ts.le(pd.Timestamp("2026-09-03T14:30Z"))]
    original = independent_candidates(old_events, close_frames)
    archived = pd.read_csv("results/reference_calendar_full_history_20260928/candidates_reference_calendars.csv")
    archived.entry_ts = pd.to_datetime(archived.entry_ts, utc=True)
    both = original.merge(archived, on=["symbol", "entry_ts"], suffixes=("_fresh", "_old"), validate="one_to_one")
    assert len(both) == len(archived) == len(original)
    assert np.allclose(both.gross_fresh, both.gross_old)
    print(f"Corrected historical candidate bridge reproduced: {len(both)}", flush=True)
    del close_frames
    summaries, monthly, priced, cache = [], [], {}, {}
    for variant in VARIANTS:
        candidates, excluded = price_candidates(events, frames, variant, cache)
        priced[variant.name] = candidates
        candidates.to_csv(OUT / f"candidates_{variant.name}.csv", index=False)
        excluded.to_csv(OUT / f"pricing_excluded_{variant.name}.csv", index=False)
        for period, (start, end) in PERIODS.items():
            for cost in (44, 68):
                tr, rejected = replay(candidates, variant, cost, start, end)
                curve = marked_curve(tr, frames, start, end)
                row = dict(variant=variant.name, period=period, cost=cost, **metrics(tr, curve))
                summaries.append(row)
                tag = f"{variant.name}_{period}_{cost}"
                tr.to_csv(OUT / f"trades_{tag}.csv", index=False)
                rejected.to_csv(OUT / f"rejected_{tag}.csv", index=False)
                curve.resample("1D").last().to_csv(OUT / f"daily_{tag}.csv")
                ends = curve.groupby(curve.index.strftime("%Y-%m")).last()
                returns = (ends/ends.shift().fillna(1)-1)*100
                monthly.extend(dict(variant=variant.name, period=period, cost=cost, month=m, return_pct=r)
                               for m, r in returns.items())
        print(f"Completed {variant.name}: {len(candidates)} priced candidates", flush=True)
    summary = pd.DataFrame(summaries)
    summary.to_csv(OUT / "summary.csv", index=False)
    pd.DataFrame(monthly).to_csv(OUT / "monthly.csv", index=False)
    selection, training_rows, fold_results = [], [], []
    combined_fold_growth = {"selected": 1., "baseline": 1.}
    boundaries = pd.to_datetime(["2026-06-01", "2026-07-01", "2026-08-01", "2026-09-01", "2026-09-26"], utc=True)
    for i, cutoff in enumerate(boundaries[:-1]):
        rows = []
        for variant in VARIANTS:
            tr, _ = replay(priced[variant.name], variant, 68, START, cutoff)
            eq = marked_curve(tr, frames, START, cutoff)
            row = dict(fold=str(cutoff), variant=variant.name, **metrics(tr, eq))
            rows.append(row)
            training_rows.append(row)
        training = pd.DataFrame(rows)
        selected = choose_rule(training, training[training.variant.eq("baseline")].iloc[0])
        selection.append(dict(train_start=str(START), train_end=str(cutoff), test_end=str(boundaries[i+1]), selected=selected))
        for label, name in (("baseline", "baseline"), ("selected", selected)):
            if name == "cash":
                row = dict(trades=0, return_pct=0., marked_dd_pct=0.)
            else:
                variant = next(v for v in VARIANTS if v.name == name)
                tr, _ = replay(priced[name], variant, 68, cutoff, boundaries[i+1])
                eq = marked_curve(tr, frames, cutoff, boundaries[i+1])
                row = metrics(tr, eq)
                tr.to_csv(OUT / f"fold_{cutoff.strftime('%Y%m')}_{label}_trades.csv", index=False)
                eq.resample("1D").last().to_csv(OUT / f"fold_{cutoff.strftime('%Y%m')}_{label}_daily.csv")
            combined_fold_growth[label] *= 1+row["return_pct"]/100
            fold_results.append(dict(fold=str(cutoff), label=label, variant=name, cost=68, **row))
    pd.DataFrame(training_rows).to_csv(OUT / "fold_training.csv", index=False)
    pd.DataFrame(selection).to_csv(OUT / "fold_selection.csv", index=False)
    pd.DataFrame(fold_results).to_csv(OUT / "fold_results.csv", index=False)
    (OUT / "manifest.json").write_text(json.dumps(dict(start=str(START), end=str(END), input_hashes=hashes,
        calendar_version=reference.VERSION, candidate_bridge_reproduced=len(both),
        protocol_sha256=hashlib.sha256(Path("docs/stock_robustness_protocol_20260928.md").read_bytes()).hexdigest(),
        fold_growth=combined_fold_growth,
        notes=["Research only; no deployment or account actions.",
               "Known baseline deadlines purge fold boundaries; all capital starts flat per fold.",
               "Next-open/gap-aware simulated fills; not reconstructed order books.",
               "Round-trip flat cost assumptions; funding added in a separate paired study.",
               "20% realized-equity sizing per stock; changing slots does not change position weight.",
               "Marked drawdown uses 5m closes, not intrabar extrema; exits inside stop bars timestamped at bar end.",
               "All samples previously inspected; retrospective chronological selection, not untouched OOS."]), indent=2))
    print(summary[(summary.cost == 68) & summary.period.isin(["historical", "recent"])].to_string(index=False))
    print("Frozen folds:\n", pd.DataFrame(fold_results).to_string(index=False))


if __name__ == "__main__":
    main()
