"""Research-only regional/technology universe ablation; never changes live config."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analysis import original_rule_study as original
from scripts.analysis.entry_rule_study import Policy
from scripts.analysis.us_only_study import load_candidates, stats
from scripts.combinations.run import _pooled_curve
from strategies.stocks.config import Config
from strategies.stocks.market import reference
from strategies.stocks.market.universe_tech import TECH
from strategies.stocks.research.backtest import Rules, run
from strategies.stocks.research.stock_categories import CATEGORY_GROUPS

OUT = Path("results/selected_tech_universe_20260928")
HISTORY = Path("results/reference_calendar_full_history_20260928")
RECENT = Path("results/reference_calendar_study_20260928")
HK = frozenset(("MINIMAX", "ZHIPU", "XIAOMI"))
KR = frozenset(("SAMSUNG", "SKHYNIX"))


def select_pools(universe, identity):
    ordinary = set(identity.loc[identity.us_ordinary, "ticker"])
    core = (set(CATEGORY_GROUPS["semiconductor_hardware"]) |
            set(CATEGORY_GROUPS["mega_cap_tech"]) | {"SKHY"})
    broad = (core | TECH | set(CATEGORY_GROUPS["ai_infrastructure_energy_space"])) - {"USAR", "HIMS", "ISRG"}
    us = {r.ticker for r in universe.itertuples() if reference.reference_market(r.instId) == "XNYS"}
    pools = {
        "reference_all": {r.ticker for r in universe.itertuples() if reference.reference_market(r.instId)},
        "us_all_hk_tech_kr_memory": us | HK | KR,
        "core_tech": (core & ordinary & us) | HK | KR,
        "broad_tech": (broad & ordinary & us) | HK | KR,
    }
    return {name: set(universe.loc[universe.ticker.isin(tickers), "instId"]) for name, tickers in pools.items()}


def record(rows, components, months, trades, curve, sample, pool, scope, cost):
    rows.append(dict(sample=sample, pool=pool, scope=scope, stock_cost_bps=cost, **stats(curve, trades)))
    tag = f"{sample}_{pool}_{scope}_{cost}"
    trades.to_csv(OUT / f"trades_{tag}.csv", index=False)
    curve.to_csv(OUT / f"equity_{tag}.csv")
    markets = trades.symbol.map(reference.reference_market).fillna("crypto")
    for market, group in trades.groupby(markets):
        components.append(dict(sample=sample, pool=pool, scope=scope, cost=cost, market=market,
                               trades=len(group), contribution_pct=(group.notional*group.net).sum()*100))
    for month, group in trades.groupby(trades.entry_ts.dt.strftime("%Y-%m")):
        months.append(dict(sample=sample, pool=pool, scope=scope, cost=cost, month=month,
                           trades=len(group), contribution_pct=(group.notional*group.net).sum()*100))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    universe = pd.read_csv("data/stocks_swap/universe.csv")
    identity = pd.read_csv("results/us_only_study_20260928/identity.csv")
    pools = select_pools(universe, identity)
    listing = universe[["instId", "ticker"]].merge(identity[["instId", "name", "etf", "us_ordinary"]], on="instId")
    listing["market"] = listing.instId.map(reference.reference_market)
    for name, ids in pools.items():
        listing[name] = listing.instId.isin(ids)
    listing.to_csv(OUT / "universe_selection.csv", index=False)
    print("Pool sizes:", {k: len(v) for k, v in pools.items()}, flush=True)
    stocks = pd.read_csv(HISTORY / "candidates_reference_calendars.csv")
    for col in ("entry_ts", "exit_ts", "original_exit_ts"):
        stocks[col] = pd.to_datetime(stocks[col], utc=True)
    policy = Policy()
    crypto = original.sleeve_select(load_candidates("crypto"), policy)
    crypto["net"] = crypto.gross-.001
    _, crypto, _ = _pooled_curve(crypto, original.DAYS, 5)
    rows, components, months = [], [], []
    for name, ids in pools.items():
        # Filter independent candidates BEFORE quotas/capacity, allowing replacements.
        selected = original.sleeve_select(stocks[stocks.symbol.isin(ids)], policy)
        for cost in (44, 68):
            for scope in ("stock_only", "combined"):
                candidates = selected.copy() if scope == "stock_only" else pd.concat([selected, crypto], ignore_index=True)
                candidates["net"] = candidates.gross-np.where(candidates.asset.eq("stock"), cost, 16 if cost == 68 else 10)/10000
                curve, trades, _ = original.pooled(candidates, policy)
                if name == "reference_all" and cost == 44:
                    expected = 33.357052 if scope == "stock_only" else 42.273988
                    assert np.isclose((curve.iloc[-1]-1)*100, expected)
                record(rows, components, months, trades, curve, "historical", name, scope, cost)
    print("Historical pools completed; corrected baseline reproduced", flush=True)
    frames, hashes = {}, {}
    for path in sorted(Path("results/us_only_study_20260928").glob("recent_market_*.csv.gz")):
        f = pd.read_csv(path)
        f["ts"] = pd.to_datetime(f.ts, utc=True)
        frames[path.name.removeprefix("recent_market_").removesuffix(".csv.gz")] = f.set_index("ts")
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    events = pd.read_csv(RECENT / "events_reference_calendars.csv")
    for col in ("event_ts", "resolve_ts", "anchor_close_ts", "event_day"):
        events[col] = pd.to_datetime(events[col], utc=True)
    days = pd.date_range("2026-09-11", "2026-09-26", tz="UTC", freq="D")
    rules = Rules(horizon="to_open", resolve_offset_minutes=60, stop_loss_bps=300,
                  max_concurrent=3, max_per_day=2, rank_column="abs_deviation", min_trailing_volume=0)
    for name, ids in pools.items():
        selected = run(events[events.inst_id.isin(ids)], frames, Config(dislocation_bps=600), rules).trades
        selected = selected.assign(symbol=selected.inst_id, asset="stock", strategy="xstock_hybrid",
                                   signal_strength=selected.signal.abs())
        for cost in (44, 68):
            selected["net"] = selected.gross-cost/10000
            curve, trades, _ = original.pooled(selected, policy, days=days)
            if name == "reference_all" and cost == 44:
                assert np.isclose((curve.iloc[-1]-1)*100, 3.997852)
            record(rows, components, months, trades, curve, "recent", name, "stock_only", cost)
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "summary.csv", index=False)
    pd.DataFrame(components).to_csv(OUT / "components.csv", index=False)
    pd.DataFrame(months).to_csv(OUT / "monthly.csv", index=False)
    sources = [HISTORY / "candidates_reference_calendars.csv", RECENT / "events_reference_calendars.csv",
               Path("results/us_only_study_20260928/identity.csv"),
               Path("results/original_rule_study_20260907/crypto_candidates.csv")]
    hashes.update({str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    (OUT / "manifest.json").write_text(json.dumps(dict(pools={k: sorted(v) for k, v in pools.items()},
        hashes=hashes, calendar_version=reference.VERSION,
        definitions={"core_tech": "Existing semiconductor/hardware and mega-cap-tech categories plus SKHY ADR; ordinary US listings only; HK MINIMAX/ZHIPU/XIAOMI; KR SAMSUNG/SKHYNIX.",
                     "broad_tech": "Core plus existing TECH and AI/infrastructure/space groups, excluding USAR/HIMS/ISRG; ordinary listings only.",
                     "us_all_hk_tech_kr_memory": "All mapped US instruments INCLUDING ETFs/preferred/crypto-linked stocks; only three HK technology names and two KR memory names."},
        caveats=["Research-only; no live universe/configuration changed.",
                 "Curated project technology taxonomy, not an exhaustive authoritative sector classification. Borderline/unreviewed names remain outside these pools.",
                 "No selection by individual historical PnL. Broad versus core is definition sensitivity, not a tuned optimal pool.",
                 "US ADR listing venue retained; SKHY and SKHYNIX can represent the same underlying. Original allocator has no underlying deduplication.",
                 "Current/frozen identity snapshot, not point-in-time security membership.",
                 "Historical independent outcomes reused with fresh admission; rules/calendars unchanged. Recent event pool repriced with fresh admission.",
                 "Historical combination retains frozen crypto signals; recent sample stock-only.",
                 "No funding, live latency, lot rounding, liquidity constraints; exact stop-barrier fills.",
                 "Daily realized equity drawdown is not marked-to-market risk. Retrospective samples, not untouched holdouts."]), indent=2))
    print(result.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
