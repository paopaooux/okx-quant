"""Conditional stock-trade attribution and earlier-exit scenarios; no orders.

Freeze actual entries, quantities and existing exits. Freed capacity does not
generate replacement trades. Classification is research metadata, not a filter.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

GROUPS = {
    "leveraged_inverse": {"CSOPSKHYNIX2L", "CSOPSAMSUNG2L", "KORU", "SKDD", "INTW"},
    "hong_kong_equity": {"MINIMAX", "ZHIPU"},
    "japan_equity": {"KIOXIA", "SOFTBANK"},
    "pre_ipo_verified": {"ANTHROPIC", "MOONSHOT"},
    "other_identity_pending": {"UNITREE", "SHEIN"},
    "us_equity": {"SMCI", "NVDA", "CRWV", "ONDS", "AMD", "IONQ", "VRT", "INTC", "PURR", "ON"},
}


def product_group(inst):
    ticker = inst.removesuffix("-USDT-SWAP")
    return next((group for group, members in GROUPS.items() if ticker in members), "unclassified")


def summarize(frame, keys):
    return frame.groupby(keys, observed=True, dropna=False).agg(
        trades=("realizedPnl", "size"), wins=("realizedPnl", lambda s: s.gt(0).sum()),
        gross=("pnl", "sum"), fees=("fee", "sum"), funding=("fundingFee", "sum"),
        net=("realizedPnl", "sum"), mean_net_bps=("net_bps", "mean"),
        median_age_s=("entry_age_s", "median"), mean_hold_h=("hold_h", "mean"))


def earlier_exit(row, desired, market, funding, slip_bps):
    """Actual exit is unchanged unless an earlier scheduled exit is possible."""
    if desired <= row.cTime:
        raise ValueError("Scheduled exit must follow entry")
    if desired >= row.uTime:
        return dict(net=row.realizedPnl, exit_ts=row.uTime, changed=False, missing=False)
    if desired not in market.index:
        return dict(net=np.nan, exit_ts=desired, changed=True, missing=True)
    side = 1 if row.direction == "long" else -1
    price = float(market.loc[desired, "open"]) * (1-side*slip_bps/10000)
    qty = row.notional / row.openAvgPx
    gross = qty * side * (price-row.openAvgPx)
    rate = -row.fee / (qty * (row.openAvgPx+row.closeAvgPx))
    fees = -rate * qty * (row.openAvgPx+price)
    # Include a settlement exactly at exit conservatively for funding debits.
    relevant = funding[(funding.inst == row.instId) & funding.ts.gt(row.cTime) & funding.ts.le(desired)]
    cash = relevant.amount.sum()
    return dict(net=gross+fees+cash, exit_ts=desired, changed=True, missing=False)


def load_market(inst):
    pieces = []
    paths = [Path("data/stocks_swap/5m") / f"{inst}.csv",
             Path("results/execution_backtest_20260911/market") / f"{inst}.csv"]
    for path in paths:
        if path.exists():
            frame = pd.read_csv(path)
            frame["ts"] = pd.to_datetime(frame.ts, utc=True)
            pieces.append(frame.set_index("ts"))
    if not pieces:
        return pd.DataFrame(columns=["open"])
    return pd.concat(pieces).loc[lambda f: ~f.index.duplicated(keep="first")].sort_index()


def run():
    out = Path("results/stock_live_diagnosis_20260928")
    out.mkdir(parents=True, exist_ok=True)
    src = Path("results/live_audit_20260928")
    trades = pd.read_csv(src / "closed_positions_enriched.csv", dtype={"posId": str})
    trades = trades[trades.asset.eq("stock")].copy()
    for col in ["cTime", "uTime", "deadline", "event_ts"]:
        trades[col] = pd.to_datetime(trades[col], utc=True, format="mixed")
    trades["group"] = trades.instId.map(product_group)
    assert not trades.group.eq("unclassified").any()
    spec = pd.read_csv("data/stocks_swap/universe.csv").set_index("instId")
    trades["notional"] = [float(spec.loc[r.instId, "ctVal"])*r.closeTotalPos*r.openAvgPx
                          for r in trades.itertuples()]
    expected = trades.notional * np.where(trades.direction.eq("long"), 1, -1) * (
        trades.closeAvgPx/trades.openAvgPx-1)
    assert np.allclose(expected, trades.pnl, atol=1e-7), "Check contract multiplier or partial closes"
    trades["net_bps"] = trades.realizedPnl / trades.notional * 10000
    trades["regime"] = np.select([
        trades.cTime.lt(pd.Timestamp("2026-09-07T07:56:41Z")),
        trades.cTime.lt(pd.Timestamp("2026-09-22T08:32:44Z"))],
        ["legacy", "policy_before_fix"], default="execution_fixed")
    trades["period"] = np.where(trades.cTime.lt(pd.Timestamp("2026-09-16T00:00:00Z")), "early", "late")
    trades["age_group"] = pd.cut(trades.entry_age_s, [-1, 120, 300, 600, np.inf],
                                 labels=["0-2m", "2-5m", "5-10m", ">10m"])
    trades["us_open"] = trades.deadline-pd.Timedelta(hours=1)
    trades["hours_to_us_open"] = (trades.us_open-trades.cTime).dt.total_seconds()/3600
    trades["wait_group"] = pd.cut(trades.hours_to_us_open, [-np.inf, 6, 12, np.inf],
                                  labels=["<=6h", "6-12h", ">12h"])
    trades["exit_class"] = np.select([
        trades.deadline.isna(), trades.late_seconds.gt(60),
        trades.late_seconds.ge(0), trades.return_bps.le(-290)],
        ["legacy_unknown", "late_over_60s", "deadline", "early_loss_stop_like"],
        default="early_other")
    # Early stop-like is an observed classification, not a proven order reason.
    trades.to_csv(out / "classified_trades.csv", index=False)
    modern = trades[trades.deadline.notna()].copy()
    for scope, frame in [("all42", trades), ("policy29", modern)]:
        for keys in [["group"], ["group", "period"], ["age_group"], ["wait_group"],
                     ["exit_class"], ["direction"], ["regime"]]:
            summarize(frame, keys).to_csv(out / f"{scope}_{'_'.join(keys)}.csv")

    filters = {
        "all": pd.Series(True, index=modern.index),
        "exclude_leveraged": modern.group.ne("leveraged_inverse"),
        "us_equity_only": modern.group.eq("us_equity"),
        "age_le_120s": modern.entry_age_s.le(120),
        "age_le_300s": modern.entry_age_s.le(300),
        "age_le_600s": modern.entry_age_s.le(600),
        "us_open_within_6h": modern.hours_to_us_open.between(0, 6),
        "us_open_within_12h": modern.hours_to_us_open.between(0, 12),
    }
    rows = []
    for name, mask in filters.items():
        for period in ["all", "early", "late"]:
            keep = modern[mask & (True if period == "all" else modern.period.eq(period))]
            rows.append(dict(filter=name, period=period, kept=len(keep), net=keep.realizedPnl.sum(),
                             mean_net_bps=keep.net_bps.mean()))
    pd.DataFrame(rows).to_csv(out / "conditional_filters.csv", index=False)

    raw = [json.loads(x["raw_json"]) for x in json.loads((src / "bills.json").read_text())]
    funding = pd.DataFrame([dict(inst=x["instId"], ts=pd.to_datetime(int(x["ts"]), unit="ms", utc=True),
                                 amount=float(x.get("pnl") or 0)) for x in raw if str(x["type"]) == "8"])
    funding.to_csv(out / "matched_funding_source.csv", index=False)
    markets = {inst: load_market(inst) for inst in modern.instId.unique()}
    for inst, frame in markets.items():
        frame.to_csv(out / f"market_{inst}.csv.gz")
    scenarios, coverage = [], []
    for row in modern.itertuples():
        market = markets[row.instId]
        matched = funding[(funding.inst == row.instId) & funding.ts.gt(row.cTime) & funding.ts.le(row.uTime)]
        assert np.isclose(matched.amount.sum(), row.fundingFee, atol=1e-7), row.instId
        targets = {"us_open": row.us_open, "cap_6h": min(row.deadline, (row.cTime+pd.Timedelta(hours=6)).ceil("5min")),
                   "cap_12h": min(row.deadline, (row.cTime+pd.Timedelta(hours=12)).ceil("5min")),
                   "on_time_deadline": row.deadline}
        for name, desired in targets.items():
            for slip in [0, 10, 30]:
                result = earlier_exit(row, desired, market, funding, slip)
                scenarios.append(dict(posId=row.posId, inst=row.instId, group=row.group, period=row.period,
                                      variant=name, slip_bps=slip, baseline=row.realizedPnl, **result))
        coverage.append(dict(inst=row.instId, entry=str(row.cTime),
                             market_start=str(market.index.min()), market_end=str(market.index.max())))
    scenarios = pd.DataFrame(scenarios)
    scenarios.to_csv(out / "earlier_exit_trades.csv", index=False)
    pd.DataFrame(coverage).to_csv(out / "market_coverage.csv", index=False)
    covered = scenarios[~scenarios.missing].copy()
    covered["delta"] = covered.net-covered.baseline
    covered.groupby(["variant", "slip_bps"]).agg(
        paired_trades=("net", "size"), changed=("changed", "sum"), baseline=("baseline", "sum"),
        counterfactual=("net", "sum"), delta=("delta", "sum")).to_csv(out / "earlier_exit_summary.csv")
    covered.groupby(["variant", "slip_bps", "group", "period"]).agg(
        paired_trades=("net", "size"), baseline=("baseline", "sum"),
        counterfactual=("net", "sum"), delta=("delta", "sum")).to_csv(out / "earlier_exit_by_group.csv")
    manifest = dict(scope="Frozen actual entries and sizes; no replacement opportunities or compounding",
        actual_trades=len(trades), policy_trades=len(modern), funding="Actual matched bills before hypothetical exit",
        prices="Scheduled-exit bar open; assumed taker execution with 0/10/30bp adverse slippage",
        caveats=["Current universe contract specifications checked against realized gross PnL.",
                 "Product metadata verified selectively for traded instruments, not a production taxonomy.",
                 "SHEIN and UNITREE deliberately left identity_pending; do not assume current IPO status.",
                 "HK/JP groups identify underlying listing geography, not a verified OKX index component calendar.",
                 "Missing scheduled-exit prices excluded with paired baseline, never forward-filled.",
                 "Existing stop timing held fixed; this is not a stop redesign backtest.",
                 "Counterfactual funding uses actual position bills; mark-price and intrasecond settlement ambiguity remain.",
                 "All thresholds and groups are exploratory on already observed data; not out-of-sample validation."])
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("All trades:\n", summarize(trades, ["group"]))
    print("New policy only:\n", summarize(modern, ["group"]))
    print("Earlier exits:\n", pd.read_csv(out / "earlier_exit_summary.csv").to_string(index=False))
    print("Conditional filters:\n", pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    run()
