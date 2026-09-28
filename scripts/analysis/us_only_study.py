"""US-listing universe ablation using archived candidates and recent candles.

Research only. Pool filtering precedes admission, so replacement candidates
compete for the original daily and concurrent stock limits.
"""
from pathlib import Path
import hashlib
import json
import re

import numpy as np
import pandas as pd

from scripts.analysis import original_rule_study as original
from scripts.analysis.entry_rule_study import Policy
from scripts.combinations.run import _pooled_curve
from strategies.stocks.config import Config
from strategies.stocks.market import data
from strategies.stocks.research import events
from strategies.stocks.research.backtest import Rules, run as stock_run

OUT = Path("results/us_only_study_20260928")
ARCHIVE = Path("results/original_rule_study_20260907")
RECENT_START = pd.Timestamp("2026-09-11T00:00:00Z")
RECENT_END = pd.Timestamp("2026-09-26T00:00:00Z")


def is_ordinary(name, etf):
    if etf != "N" or re.search(r"preferred|warrant|\bunits?\b|\brights\b|notes due", name, re.I):
        return False
    return bool(re.search(r"common|ordinary|depositary|depository|\bADS\b|registered shares|registry shares|voting shares", name, re.I))


def classify(universe):
    directory = {}
    for filename, key, exchange in [("nasdaqlisted.txt", "Symbol", "NASDAQ"),
                                     ("otherlisted.txt", "ACT Symbol", "OTHER_US")]:
        frame = pd.read_csv(OUT / filename, sep="|")
        for row in frame.to_dict("records"):
            if row.get("Test Issue") != "N":
                continue
            directory[row[key]] = dict(name=row["Security Name"], etf=row["ETF"], exchange=exchange)
    records = []
    for row in universe.itertuples():
        symbol = {"BRKB": "BRK.B"}.get(row.ticker, row.ticker)
        found = directory.get(symbol)
        # The exchange directory abbreviates TSM's name without its ADR type.
        # Identity verified against investor.tsmc.com stock information.
        ordinary = bool(found and (is_ordinary(found["name"], found["etf"]) or
                                   (symbol == "TSM" and found["etf"] == "N")))
        records.append(dict(instId=row.instId, ticker=row.ticker, us_symbol=symbol,
                            matched=bool(found), us_ordinary=ordinary,
                            us_equity_or_etf=bool(found and (ordinary or found["etf"] == "Y")),
                            **(found or dict(name="unverified", etf="", exchange=""))))
    return pd.DataFrame(records)


def load_candidates(name):
    frame = pd.read_csv(ARCHIVE / f"{name}_candidates.csv")
    for col in ["entry_ts", "exit_ts", "early_exit_ts", "original_exit_ts"]:
        frame[col] = pd.to_datetime(frame[col], utc=True, format="mixed")
    return frame


def stats(curve, trades):
    return dict(trades=len(trades), return_pct=float((curve.iloc[-1]-1)*100),
                realized_daily_dd_pct=float((curve/curve.cummax()-1).min()*100),
                win_pct=float(trades.net.gt(0).mean()*100) if len(trades) else None,
                mean_net_bps=float(trades.net.mean()*10000) if len(trades) else None)


def record(summary, components, trades, curve, sample, pool, cost, scope, baseline_keys):
    keys = set(zip(trades.entry_ts, trades.symbol))
    row = dict(sample=sample, pool=pool, round_trip_stock_bps=cost, scope=scope,
               replacement_trades=len(keys-baseline_keys) if baseline_keys is not None else 0,
               **stats(curve, trades))
    summary.append(row)
    trades.to_csv(OUT / f"trades_{sample}_{scope}_{pool}_{cost}.csv", index=False)
    curve.to_csv(OUT / f"equity_{sample}_{scope}_{pool}_{cost}.csv")
    for asset, group in trades.groupby("asset"):
        components.append(dict(sample=sample, scope=scope, pool=pool, cost=cost, asset=asset,
                               trades=len(group), contribution_pct=float((group.notional*group.net).sum()*100)))
    return keys


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    universe = pd.read_csv("data/stocks_swap/universe.csv")
    identity = classify(universe)
    identity.to_csv(OUT / "identity.csv", index=False)
    pools = {"all": set(universe.instId),
             "us_ordinary": set(identity.loc[identity.us_ordinary, "instId"]),
             "us_with_etf": set(identity.loc[identity.us_equity_or_etf, "instId"])}
    print("Pools:", {k: len(v) for k, v in pools.items()}, flush=True)
    print("Unverified:", identity.loc[~identity.matched, "ticker"].tolist(), flush=True)
    policy = Policy()
    stocks, crypto = load_candidates("stock"), load_candidates("crypto")
    quant = original.sleeve_select(crypto, policy)
    quant["net"] = quant.gross-.001
    _, quant, _ = _pooled_curve(quant, original.DAYS, 5)
    summary, components, monthly = [], [], []
    baseline_keys = {}
    for pool, ids in pools.items():
        selected = original.sleeve_select(stocks[stocks.symbol.isin(ids)], policy)
        for cost in [10, 30, 44, 68]:
            for scope in ["stock_only", "combined"]:
                candidates = selected.copy() if scope == "stock_only" else pd.concat([selected, quant], ignore_index=True)
                candidates["net"] = candidates.gross-np.where(candidates.asset.eq("stock"), cost, 16 if cost == 68 else 10)/10000
                curve, trades, _ = original.pooled(candidates, policy)
                if pool == "all" and cost == 44 and scope == "combined":
                    assert len(trades) == 299
                    assert np.isclose((curve.iloc[-1]-1)*100, 50.277153751470195)
                key = (cost, scope)
                keys = record(summary, components, trades, curve, "historical", pool, cost, scope, baseline_keys.get(key))
                if pool == "all":
                    baseline_keys[key] = keys
                for month, group in trades.groupby(trades.entry_ts.dt.strftime("%Y-%m")):
                    monthly.append(dict(sample="historical", pool=pool, scope=scope, cost=cost, month=month,
                                        trades=len(group), mean_net_bps=group.net.mean()*10000,
                                        contribution_pct=(group.notional*group.net).sum()*100))
    print("Historical baseline reproduced; universe substitutions finished", flush=True)

    # The rolling live cache no longer contains the full September window.
    frames = data.to_bar_end(data.load_panel(sorted(universe.instId), "5m", "data/stocks_swap"), "5m")
    frames = {inst: frame.loc[frame.index <= RECENT_END] for inst, frame in frames.items()}
    config = Config(dislocation_bps=600, fee_bps=10, slippage_bps=12)
    raw_events = events.off_hours_dislocation(frames, config)
    raw_events = raw_events[raw_events.event_ts.ge(RECENT_START) & raw_events.event_ts.lt(RECENT_END)]
    raw_events.to_csv(OUT / "recent_events.csv", index=False)
    coverage = []
    for inst, frame in frames.items():
        frame.to_csv(OUT / f"recent_market_{inst}.csv.gz")
        coverage.append(dict(inst=inst, first=str(frame.index.min()), last=str(frame.index.max()), rows=len(frame)))
    pd.DataFrame(coverage).to_csv(OUT / "recent_coverage.csv", index=False)
    rules = Rules(horizon="to_open", resolve_offset_minutes=60, stop_loss_bps=300,
                  max_concurrent=3, max_per_day=2, rank_column="abs_deviation", min_trailing_volume=0)
    days = pd.date_range(RECENT_START.normalize(), RECENT_END.normalize(), freq="D")
    recent_baseline = {}
    for pool, ids in pools.items():
        result = stock_run(raw_events[raw_events.inst_id.isin(ids)], frames, config, rules)
        candidates = result.trades.assign(symbol=lambda f: f.inst_id, asset="stock", strategy="xstock_hybrid",
                                          signal_strength=lambda f: f.signal.abs())
        for cost in [10, 30, 44, 68]:
            candidates["net"] = candidates.gross-cost/10000
            curve, trades, _ = original.pooled(candidates, policy, days=days)
            keys = record(summary, components, trades, curve, "recent", pool, cost, "stock_only", recent_baseline.get(cost))
            if pool == "all":
                recent_baseline[cost] = keys
    actual = pd.read_csv("results/stock_live_diagnosis_20260928/classified_trades.csv")
    actual_rows = []
    for pool, ids in pools.items():
        for cohort in ["all42", "policy29"]:
            subset = actual[actual.instId.isin(ids) & (True if cohort == "all42" else actual.regime.ne("legacy"))]
            actual_rows.append(dict(pool=pool, cohort=cohort, trades=len(subset), net=subset.realizedPnl.sum()))
    pd.DataFrame(actual_rows).to_csv(OUT / "actual_filter.csv", index=False)
    pd.DataFrame(summary).to_csv(OUT / "summary.csv", index=False)
    pd.DataFrame(components).to_csv(OUT / "components.csv", index=False)
    pd.DataFrame(monthly).to_csv(OUT / "historical_monthly.csv", index=False)
    manifest = dict(pools={k: sorted(v) for k, v in pools.items()}, historical_start=str(original.START),
        historical_end=str(original.END), recent_start=str(RECENT_START), recent_end=str(RECENT_END),
        archive_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in [ARCHIVE / "stock_candidates.csv", ARCHIVE / "crypto_candidates.csv",
                                  OUT / "nasdaqlisted.txt", OUT / "otherlisted.txt"]},
        definition="US-exchange-listed common/ordinary equity and ADRs; US incorporation not required",
        baseline_assertion="299 trades and +50.27715375147% original combined result reproduced",
        caveats=["Current listing directory, NOT a point-in-time historical security master; survivorship/listing bias possible.",
                 "Ticker match does not independently verify every OKX index component; uncertain identities excluded.",
                 "Historical prices rolled out of live cache: archived independent candidate outcomes reused, not newly repriced.",
                 "Historical original two-stage stock/crypto admission retained; not identical to unified live allocator.",
                 "Recent rerun stock-only; crypto occupancy and live signal/data delays not modeled.",
                 "Bar-close entries and exact stop barriers retained from original engine; gaps can worsen execution.",
                 "Round-trip cost assumptions 10/30/44/68bp; 10bp is fee-only, not measured total cost. Funding, lot rounding and market impact not separately modeled.",
                 "Realized daily drawdown excludes intraday and open-position mark-to-market risk.",
                 "Actual filtered PnL has no replacement entries; simulated universes rerun admission with replacements.",
                 "All data retrospective; no result is an untouched out-of-sample guarantee."])
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(pd.DataFrame(summary).to_string(index=False))
    print("Actual subset:", pd.DataFrame(actual_rows).to_string(index=False))


if __name__ == "__main__":
    main()
