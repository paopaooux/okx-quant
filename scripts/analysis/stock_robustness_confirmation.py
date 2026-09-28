"""Post-screen execution stress for fixed candidates, not a new threshold search."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pandas as pd

from scripts.analysis.stock_funding_robustness import OUT as FUNDED, PERIODS as FUND_PERIODS, load_candidates
from scripts.analysis.stock_robustness_study import (
    OUT as STUDY, PERIODS, VARIANTS, load_frames, marked_curve, metrics, replay,
)

OUT = Path("results/stock_robustness_confirmation_20260928")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frames, _, _, hashes = load_frames()
    rows = []
    for scenario in ("unfunded", "funded_conservative"):
        periods = {k: PERIODS[k] for k in ("historical", "recent")} if scenario == "unfunded" else FUND_PERIODS
        root = STUDY if scenario == "unfunded" else FUNDED
        for delay, candidate_name in ((0, "baseline"), (5, "delay_5m"), (10, "delay_10m")):
            path = root / f"candidates_{candidate_name}.csv"
            hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            candidates = load_candidates(path)
            if scenario != "unfunded":
                candidates["funding_events"] = candidates.funding_events.map(json.loads)
            for name in ("baseline", "displacement_le_8pct", "stock_cap_2"):
                variant = replace(next(v for v in VARIANTS if v.name == name), delay_bars=delay//5)
                for period, (start, end) in periods.items():
                    for cost in (44, 68):
                        tr, rejected = replay(candidates, variant, cost, start, end)
                        eq = marked_curve(tr, frames, start, end)
                        rows.append(dict(variant=name, delay_minutes=delay, scenario=scenario,
                                         period=period, cost=cost, **metrics(tr, eq)))
                        tag = f"{name}_{delay}_{scenario}_{period}_{cost}"
                        tr.to_csv(OUT / f"trades_{tag}.csv", index=False)
                        rejected.to_csv(OUT / f"rejected_{tag}.csv", index=False)
                        eq.resample("1D").last().to_csv(OUT / f"daily_{tag}.csv")
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "summary.csv", index=False)
    (OUT / "manifest.json").write_text(json.dumps(dict(input_hashes=hashes,
        followup_protocol_sha256=hashlib.sha256(Path("docs/stock_robustness_followup_20260928.md").read_bytes()).hexdigest(),
        caveats=["Follow-up rules chosen after viewing parent results; no new OOS evidence.",
                 "Same 8% threshold and two-slot rule, only already declared fill delays vary.",
                 "These minutes are scenarios, not measurements of the live bot's delay distribution.",
                 "Delay variants share preknown baseline deadlines and original observed signal displacement.",
                 "All causal pricing, coverage and funding limitations of parent studies apply."]), indent=2))
    print(result[(result.cost == 68) & result.period.isin(["historical", "july_september", "recent"])].to_string(index=False))


if __name__ == "__main__":
    main()
