"""Paired execution bridge, preserving candidate identity and admission rules."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analysis import original_rule_study as original
from scripts.analysis.entry_rule_study import Policy
from scripts.analysis.stock_funding_robustness import load_candidates
from scripts.analysis.stock_robustness_study import (
    OUT as STUDY, PERIODS, STEP, VARIANTS, load_frames, marked_curve, metrics, replay,
)
from strategies.stocks.research.backtest import _exit_scan

OUT = Path("results/stock_execution_audit_20260928")
STAGES = ("signal_close_exact_stop", "next_open_exact_stop",
          "next_open_deadline_open", "next_open_gap_stop")


def execution_stages(candidates, frames):
    stages = {name: [] for name in STAGES}
    close_frames = {k: f.set_axis(f.index+STEP) for k, f in frames.items()}
    for row in candidates.to_dict("records"):
        f = frames[row["symbol"]]
        for name in STAGES:
            item = row.copy()
            if name != "next_open_gap_stop":
                ep = row["signal_price"] if name == STAGES[0] else row["entry_price"]
                stamp, px, why = _exit_scan(close_frames[row["symbol"]], row["entry_ts"],
                                            ep, row["side"], row["deadline"], .03, 0.)
                if name == "next_open_deadline_open" and why == "deadline":
                    px = float(f.loc[row["deadline"], "open"])
                item.update(entry_price=ep, exit_price=px, exit_ts=stamp, reason=why,
                            gross=row["side"]*(px/ep-1),
                            hold_h=(stamp-row["entry_ts"]).total_seconds()/3600)
            stages[name].append(item)
    return {name: pd.DataFrame(rows) for name, rows in stages.items()}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frames, _, _, hashes = load_frames()
    candidates = load_candidates(STUDY / "candidates_baseline.csv")
    stages = execution_stages(candidates, frames)
    old_path = Path("results/reference_calendar_full_history_20260928/candidates_reference_calendars.csv")
    archived = pd.read_csv(old_path)
    for col in ("entry_ts", "exit_ts", "original_exit_ts"):
        archived[col] = pd.to_datetime(archived[col], utc=True)
    regenerated = stages[STAGES[0]]
    regenerated = regenerated[regenerated.entry_ts.le(original.END)]
    both = regenerated.merge(archived, on=["symbol", "entry_ts"], validate="one_to_one", suffixes=("_new", "_old"))
    assert len(both) == len(archived) == len(regenerated)
    np.testing.assert_allclose(both.gross_new, both.gross_old, atol=1e-12)
    np.testing.assert_allclose(both.entry_price_new, both.entry_price_old, atol=1e-12)
    assert both.exit_ts_new.eq(both.exit_ts_old).all()
    rows = []
    previous = pd.read_csv("results/reference_calendar_full_history_20260928/summary.csv")
    for cost in (44, 68):
        tr = original.sleeve_select(archived, Policy())
        tr["net"] = tr.gross-cost/10000
        curve, tr, _ = original.pooled(tr, Policy())
        expected = previous[(previous.variant == "reference_calendars") &
                            (previous.scope == "stock_only") & (previous.stock_cost_bps == cost)].iloc[0]
        assert len(tr) == expected.trades
        assert np.isclose((curve.iloc[-1]-1)*100, expected.return_pct)
        rows.append(dict(stage="archived_reported", period="historical", cost=cost,
                         trades=len(tr), return_pct=(curve.iloc[-1]-1)*100,
                         win_pct=tr.net.gt(0).mean()*100))
    for name, pool in stages.items():
        pool.to_csv(OUT / f"candidates_{name}.csv", index=False)
        for period in ("historical", "recent"):
            start, end = PERIODS[period]
            for cost in (44, 68):
                tr, _ = replay(pool, VARIANTS[0], cost, start, end)
                curve = marked_curve(tr, frames, start, end)
                rows.append(dict(stage=name, period=period, cost=cost, **metrics(tr, curve)))
                tr.to_csv(OUT / f"trades_{name}_{period}_{cost}.csv", index=False)
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "summary.csv", index=False)
    pair = candidates[["symbol", "entry_ts", "deviation", "signal_volume", "volume_24h"]].copy()
    for name, pool in stages.items():
        assert pool[["symbol", "entry_ts"]].equals(pair[["symbol", "entry_ts"]])
        for col in ("gross", "entry_price", "exit_price", "exit_ts", "reason"):
            pair[f"{name}_{col}"] = pool[col].to_numpy()
    pair["entry_delta_bps"] = (pair.next_open_exact_stop_gross-pair.signal_close_exact_stop_gross)*10000
    pair["deadline_delta_bps"] = (pair.next_open_deadline_open_gross-pair.next_open_exact_stop_gross)*10000
    pair["gap_delta_bps"] = (pair.next_open_gap_stop_gross-pair.next_open_deadline_open_gross)*10000
    pair["total_delta_bps"] = (pair.next_open_gap_stop_gross-pair.signal_close_exact_stop_gross)*10000
    pair.sort_values("total_delta_bps").to_csv(OUT / "candidate_deltas.csv", index=False)
    same_book = []
    for period in ("historical", "recent"):
        start, end = PERIODS[period]
        admitted, _ = replay(stages[STAGES[0]], VARIANTS[0], 68, start, end)
        pairs = admitted[["symbol", "entry_ts", "notional"]].merge(pair, on=["symbol", "entry_ts"], validate="one_to_one")
        pairs.to_csv(OUT / f"fixed_book_deltas_{period}.csv", index=False)
        for field in ("entry", "deadline", "gap", "total"):
            same_book.append(dict(period=period, component=field,
                contribution_delta_pct=(pairs.notional*pairs[f"{field}_delta_bps"]/100).sum(),
                changed=int(pairs[f"{field}_delta_bps"].abs().gt(1e-7).sum())))
    pd.DataFrame(same_book).to_csv(OUT / "fixed_book_summary.csv", index=False)
    (OUT / "manifest.json").write_text(json.dumps(dict(
        old_candidates_sha256=hashlib.sha256(old_path.read_bytes()).hexdigest(),
        reproduced_candidates=len(both), input_hashes=hashes,
        method="Sequential one-assumption changes; all stages replay identical common-deadline candidate admission.",
        caveats=["Archived reported row uses the former actual-exit boundary rule; all other rows use common preknown deadlines.",
                 "Portfolio differences include admission/compounding feedback; fixed-book deltas hold old admitted trades/notionals fixed.",
                 "Fixed-book deltas are execution attribution, not a feasible portfolio with changed exit times.",
                 "All fills remain OHLC assumptions; funding excluded in this bridge.",
                 "Five-minute stop timestamps are end-of-bar approximations."]), indent=2))
    print(summary.to_string(index=False))
    print(pd.DataFrame(same_book).to_string(index=False))


if __name__ == "__main__":
    main()
