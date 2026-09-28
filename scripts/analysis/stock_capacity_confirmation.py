"""Post-screen shared-capacity tests; stock funding, delay and allocator controls."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analysis.stock_funding_robustness import OUT as FUNDED, load_candidates
from scripts.analysis.stock_portfolio_robustness import (
    CRYPTO_STEP, admit, drawdown_durations, occupancy, portfolio_curve, prepare_crypto,
)
from scripts.analysis.stock_robustness_study import (
    OUT as STUDY, PERIODS, VARIANTS, load_frames, metrics,
)

OUT = Path("results/stock_capacity_confirmation_20260928")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    stocks, _, _, hashes = load_frames()
    start, end = PERIODS["historical"]
    path = Path("results/original_rule_study_20260907/crypto_candidates.csv")
    hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    crypto = prepare_crypto(pd.read_csv(path))
    crypto = crypto[crypto.entry_ts.ge(start) & crypto.entry_ts.lt(end)].copy()
    crypto_frames = {}
    for symbol in crypto.symbol.unique():
        path = Path("data/klines") / f"{symbol}.csv.gz"
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        frame = pd.read_csv(path, usecols=["ts", "open", "close"])
        frame.index = pd.to_datetime(frame.ts, unit="ms", utc=True)
        frame = frame[(frame.index >= start-pd.Timedelta(days=1)) & (frame.index <= end)].sort_index()
        assert not frame.index.has_duplicates
        assert not len(pd.date_range(frame.index.min(), frame.index.max(), freq=CRYPTO_STEP).difference(frame.index))
        sub = crypto[crypto.symbol.eq(symbol)]
        np.testing.assert_allclose(frame.open.reindex(pd.DatetimeIndex(sub.entry_ts)), sub.entry_price)
        crypto_frames[symbol] = frame
    windows = [("historical_unfunded", STUDY, start, end),
               ("july_august_stock_funded", FUNDED, pd.Timestamp("2026-07-01T00:00Z"), pd.Timestamp("2026-09-01T00:00Z"))]
    rows, months, components = [], [], []
    for scenario, root, a, b in windows:
        for delay, name in ((0, "baseline"), (5, "delay_5m"), (10, "delay_10m")):
            path = root / f"candidates_{name}.csv"
            hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            stock = load_candidates(path).assign(asset="stock")
            if root == STUDY:
                stock["funding"] = 0.
                stock["funding_events"] = [[] for _ in range(len(stock))]
            else:
                stock["funding_events"] = stock.funding_events.map(json.loads)
            all_candidates = pd.concat([stock, crypto], ignore_index=True)
            stock_sleeve, _ = admit(stock, VARIANTS[0], 68, a, b)
            crypto_sleeve, _ = admit(crypto, VARIANTS[0], 68, a, b)
            preselected = pd.concat([stock_sleeve, crypto_sleeve], ignore_index=True)
            for slots in range(1, 6):
                for allocation, pool in (("unified", all_candidates), ("two_stage", preselected)):
                    tr, rejected = admit(pool, VARIANTS[0], 68, a, b, slots=slots, enforce_sleeves=allocation == "unified")
                    curve = portfolio_curve(tr, stocks, crypto_frames, a, b)
                    tag = f"{scenario}_{delay}_{slots}_{allocation}"
                    rows.append(dict(scenario=scenario, delay_minutes=delay, slots=slots, allocation=allocation,
                                     **metrics(tr, curve), **occupancy(tr, a, b), **drawdown_durations(curve)))
                    tr.to_csv(OUT / f"trades_{tag}.csv", index=False)
                    rejected.to_csv(OUT / f"rejected_{tag}.csv", index=False)
                    curve.resample("1D").last().to_csv(OUT / f"daily_{tag}.csv")
                    ends = curve.groupby(curve.index.strftime("%Y-%m")).last()
                    for month, change in ((ends/ends.shift().fillna(1)-1)*100).items():
                        months.append(dict(scenario=scenario, delay_minutes=delay, slots=slots,
                                           allocation=allocation, month=month, return_pct=change))
                    for asset, group in tr.groupby("asset"):
                        components.append(dict(scenario=scenario, delay_minutes=delay, slots=slots,
                            allocation=allocation, asset=asset, trades=len(group),
                            contribution_pct=(group.notional*group.net).sum()*100))
        print(f"Capacity confirmation finished: {scenario}", flush=True)
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "summary.csv", index=False)
    pd.DataFrame(months).to_csv(OUT / "monthly.csv", index=False)
    pd.DataFrame(components).to_csv(OUT / "components.csv", index=False)
    parent = pd.read_csv("results/stock_portfolio_robustness_20260928/capacity_summary.csv")
    parent = parent.query("scope == 'shared' and sizing == 'fixed_20pct' and cost == 68")
    fresh = result.query("scenario == 'historical_unfunded' and delay_minutes == 0 and allocation == 'unified'")
    matched = parent.merge(fresh, on="slots", validate="one_to_one", suffixes=("_old", "_new"))
    assert len(matched) == 5
    np.testing.assert_allclose(matched.return_pct_old, matched.return_pct_new)
    np.testing.assert_array_equal(matched.trades_old, matched.trades_new)
    (OUT / "manifest.json").write_text(json.dumps(dict(input_hashes=hashes,
        protocol_sha256=hashlib.sha256(Path("docs/stock_capacity_followup_20260928.md").read_bytes()).hexdigest(),
        costs=dict(stock=68, crypto=16), position_weight=.2,
        caveats=["Post-screen retrospective check; all five capacities reported, no rule selected or deployed.",
                 "Only stock entries are delayed; crypto remains a frozen Binance-label comparator.",
                 "Stock-only funding in covered July-August window; crypto funding omitted.",
                 "Two-stage preselection and unified books share known horizons and causal entry clocks.",
                 "All OHLC, sizing, crypto missing-feature and historical-data limitations remain."]), indent=2))
    print(result[result.slots.isin([2, 5])].to_string(index=False))


if __name__ == "__main__":
    main()
