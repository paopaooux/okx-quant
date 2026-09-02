"""Prove the panel contains no lookahead, mechanically rather than by argument.

The test is truncation.  Rebuild the panel from a data set that has been cut off
at bar T and compare the surviving rows to the full build.  A feature at bar t
that secretly reads bar t+1 will change when the tail is removed; one that only
reads history cannot.  Labels get the same treatment with the barrier horizon
added back, since a label at bar t is *supposed* to see up to t+H+1.

Any mismatch here invalidates every number downstream, so this runs before
training, not after.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.data import build

SYM = "ETHUSDT"
CUT = 60_000


def main() -> None:
    full = build.build_symbol(SYM, None)
    trunc = build.build_symbol(SYM, None, limit=CUT)

    feat_cols = [c for c in full.columns
                 if c not in ("symbol", "dt", "ts", "entry_px")
                 and not c.startswith(("long_win_", "short_win_", "held_", "exit_ret_"))]
    max_h = max(h for _, _, h in build.CONFIGS)

    a = full[full["ts"].isin(trunc["ts"])].set_index("ts").sort_index()
    b = trunc.set_index("ts").sort_index()
    assert len(a) == len(b), f"row mismatch {len(a)} vs {len(b)}"

    print(f"comparing {len(b):,} rows of {SYM} rebuilt from a data set cut at bar {CUT:,}\n")

    bad = []
    # Features: every row of the truncated build must match, including its last.
    for c in feat_cols:
        x, y = a[c].to_numpy(float), b[c].to_numpy(float)
        both_nan = np.isnan(x) & np.isnan(y)
        diff = np.abs(x - y)
        diff[both_nan] = 0.0
        n = int(np.nansum(diff > 1e-9)) + int((np.isnan(x) ^ np.isnan(y)).sum())
        if n:
            bad.append((c, n, float(np.nanmax(diff))))
    print(f"features checked: {len(feat_cols)}   mismatching: {len(bad)}")
    for c, n, d in bad:
        print(f"   LOOKAHEAD  {c}: {n} rows differ, max |delta| {d:.3e}")

    # Labels: drop the last max_h+2 rows, which legitimately lack their future.
    lab_cols = [c for c in full.columns if c.startswith(("long_win_", "short_win_"))]
    keep = b.index[:-(max_h + 2)]
    lbad = []
    for c in lab_cols:
        x = a.loc[keep, c].to_numpy(float)
        y = b.loc[keep, c].to_numpy(float)
        n = int(np.nansum(np.abs(x - y) > 0))
        if n:
            lbad.append((c, n))
    print(f"labels checked:   {len(lab_cols)}   mismatching: {len(lbad)}")
    for c, n in lbad:
        print(f"   LABEL REACH  {c}: {n} rows differ")

    # The converse, and the sharper test: a label must actually *use* its future.
    # Amputate the future at many random points and check that exactly the
    # non-zero labels in the amputated tail flip, and no others.  A single cut
    # point is not enough -- labels are strongly autocorrelated, so one quiet
    # stretch can show zero flips and mean nothing.
    kl = pd.read_csv(build.DATA / "klines" / f"{SYM}.csv.gz")
    o, hi, lo, cl = (kl[x].to_numpy(float) for x in ("open", "high", "low", "close"))
    sig = pd.Series(np.log(cl)).diff().ewm(span=build.VOL_SPAN, adjust=False).std().shift(1).to_numpy()
    rng = np.random.default_rng(3)
    print()
    for name, kk, hh in build.CONFIGS:
        w = kk * sig * np.sqrt(hh)
        lf, sf, _, _ = build.triple_barrier(o, hi, lo, w, hh)
        flipped = nonzero = cells = 0
        for n in rng.integers(20_000, len(o) - 2_000, 25):
            n = int(n)
            lc, sc, _, _ = build.triple_barrier(o[:n], hi[:n], lo[:n], w[:n], hh)
            t = np.arange(n - hh, n)
            flipped += int((lf[t] != lc[t]).sum()) + int((sf[t] != sc[t]).sum())
            nonzero += int((lf[t] != 0).sum()) + int((sf[t] != 0).sum())
            cells += 2 * len(t)
        mark = "OK" if flipped == nonzero else "MISMATCH"
        print(f"  horizon use {name} H={hh:2d}: {flipped}/{cells} tail cells flip, "
              f"{nonzero} true non-zero labels there -> {mark}")
        if flipped != nonzero:
            bad.append((f"horizon_use_{name}", abs(flipped - nonzero), float("nan")))

    ok = not bad and not lbad
    print("\n" + ("PASS - no lookahead detected" if ok else "FAIL - fix before training"))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
