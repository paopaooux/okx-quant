"""Corrected stocks plus first-observed live crypto sides, not actual account NAV."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analysis.live_month_replay import replay_exit
from scripts.analysis.shared_two_slot_weights import (
    SHARPE_PROTOCOL, WEIGHTS, capital_usage, daily_risk, resize_book,
)
from scripts.analysis.stock_funding_robustness import OUT as FUNDED, load_candidates
from scripts.analysis.stock_portfolio_robustness import (
    CRYPTO_STEP, admit, drawdown_durations, portfolio_curve,
)
from scripts.analysis.stock_robustness_study import VARIANTS, load_frames, metrics

OUT = Path("results/september_shared_weights_20260928")
LIVE = Path("results/live_month_replay_20260928")
AUDIT = Path("results/live_audit_20260928")
START, END = pd.Timestamp("2026-09-04T00:00Z"), pd.Timestamp("2026-09-26T00:00Z")
CRYPTO_COLUMNS = ["symbol", "asset", "entry_ts", "exit_ts", "coverage_deadline",
    "entry_price", "exit_price", "gross", "funding", "funding_events", "side",
    "signal_strength", "hold_h", "source_i", "base_free_i", "observed_at",
    "bar_at", "p_up", "width", "reason", "ambiguous"]


def crypto_candidates(predictions, markets, start, end):
    frame = predictions.copy()
    for col in ("observed_at", "bar_at"):
        frame[col] = pd.to_datetime(frame[col], utc=True)
    if frame.duplicated(["symbol", "bar_at"]).any():
        raise ValueError("Expected frozen first-observation predictions, not same-bar revisions")
    frame["entry_ts"] = frame.observed_at.dt.ceil("15min")
    frame["coverage_deadline"] = frame.bar_at+49*CRYPTO_STEP
    frame["age_s"] = (frame.observed_at-frame.bar_at).dt.total_seconds()
    frame["side"] = frame.signal_side.map({"long": 1, "short": -1}).fillna(0)
    rows, excluded = [], []
    eligible = frame[frame.side.ne(0)].sort_values(["entry_ts", "symbol", "bar_at"])
    for i, row in enumerate(eligible.itertuples()):
        reason = None
        if row.entry_ts < start or row.coverage_deadline >= end:
            reason = "outside_known_horizon"
        elif not 900 <= row.age_s <= 1800:
            reason = "unclosed_or_stale_bar"
        elif not np.isfinite(row.width) or not 0 < row.width < 1:
            reason = "invalid_width"
        elif not np.isfinite(row.p_up) or not 0 <= row.p_up <= 1:
            reason = "invalid_probability"
        elif row.entry_ts >= row.coverage_deadline:
            reason = "expired"
        elif row.symbol not in markets:
            reason = "missing_market"
        outcome = None if reason else replay_exit(markets[row.symbol], row.entry_ts,
            row.coverage_deadline, row.side, row.width, slip_bps=0)
        if not reason and outcome is None:
            reason = "incomplete_market_window"
        if reason:
            excluded.append(dict(symbol=row.symbol, observed_at=row.observed_at, reason=reason))
            continue
        rows.append(dict(symbol=row.symbol, asset="crypto", entry_ts=row.entry_ts,
            exit_ts=outcome["exit_ts"], coverage_deadline=row.coverage_deadline,
            entry_price=outcome["entry_px"], exit_price=outcome["exit_px"],
            gross=row.side*(outcome["exit_px"]/outcome["entry_px"]-1),
            funding=0., funding_events=[], side=int(row.side),
            signal_strength=abs(row.p_up-.5),
            hold_h=(outcome["exit_ts"]-row.entry_ts).total_seconds()/3600,
            # Adapt unique live bars to the allocator without a label-horizon cooldown.
            source_i=i, base_free_i=i+1, observed_at=row.observed_at, bar_at=row.bar_at,
            p_up=row.p_up, width=row.width, reason=outcome["reason"], ambiguous=outcome["ambiguous"]))
    return pd.DataFrame(rows, columns=CRYPTO_COLUMNS), pd.DataFrame(
        excluded, columns=["symbol", "observed_at", "reason"])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    stocks, _, stock_coverage, hashes = load_frames()
    stock_coverage.to_csv(OUT / "stock_market_coverage.csv", index=False)

    def record(path):
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        return path

    predictions = pd.read_csv(record(LIVE / "predictions.csv.gz"))
    markets, market_coverage = {}, []
    for symbol in sorted(predictions.symbol.unique()):
        frame = pd.read_csv(record(LIVE / f"market_{symbol}.csv.gz"))
        frame.index = pd.to_datetime(frame.ts, utc=True)
        frame = frame.sort_index()
        if frame.index.has_duplicates:
            raise ValueError(f"Duplicate crypto candles: {symbol}")
        expected = pd.date_range(START-CRYPTO_STEP, END, freq=CRYPTO_STEP)
        missing = len(expected.difference(frame.index))
        market_coverage.append(dict(symbol=symbol, first=str(frame.index.min()),
            last=str(frame.index.max()), missing_bars=missing))
        if missing or not np.isfinite(frame[["open", "high", "low", "close"]]).all().all():
            raise ValueError(f"Incomplete or invalid crypto marks: {symbol}")
        markets[symbol] = frame
    pd.DataFrame(market_coverage).to_csv(OUT / "crypto_market_coverage.csv", index=False)
    crypto, excluded = crypto_candidates(predictions, markets, START, END)
    crypto.to_csv(OUT / "crypto_candidates.csv", index=False)
    excluded.to_csv(OUT / "crypto_coverage_excluded.csv", index=False)
    coverage = pd.read_csv(record(LIVE / "daily_signal_coverage.csv"))
    stamps = pd.to_datetime(coverage.observed_at, utc=True)
    coverage[stamps.ge(START) & stamps.lt(END)].to_csv(OUT / "daily_signal_coverage.csv", index=False)

    # Check entry/exit identities against the original crypto-only live-side replay.
    control, _ = admit(crypto, VARIANTS[0], 68, START, END, slots=5)
    archived = pd.read_csv(record(LIVE / "trades_recorded_0.csv"))
    for col in ("entry_ts", "exit_ts"):
        archived[col] = pd.to_datetime(archived[col], utc=True)
    archived = archived[archived.entry_ts.ge(START) & archived.exit_ts.lt(END)].reset_index(drop=True)
    pd.testing.assert_frame_equal(control[["symbol", "entry_ts", "exit_ts"]].reset_index(drop=True),
                                  archived[["symbol", "entry_ts", "exit_ts"]])
    np.testing.assert_allclose(control.entry_price, archived.entry_px)
    np.testing.assert_allclose(control.exit_price, archived.exit_px)
    np.testing.assert_allclose(control.gross, archived.gross_return)
    control.to_csv(OUT / "crypto_only_identity_control.csv", index=False)

    summaries, components = [], []
    for delay, name in ((0, "baseline"), (5, "delay_5m"), (10, "delay_10m")):
        stock = load_candidates(record(FUNDED / f"candidates_{name}.csv")).assign(asset="stock")
        stock["funding_events"] = stock.funding_events.map(json.loads)
        for scope, pool in (("stock_only_paired", stock),
                            ("shared_live_crypto", pd.concat([stock, crypto], ignore_index=True))):
            book, rejected = admit(pool, VARIANTS[0], 68, START, END, slots=2)
            rejected.to_csv(OUT / f"rejected_{scope}_{delay}.csv", index=False)
            for cost in (44, 68):
                for weight in WEIGHTS:
                    trades = resize_book(book, weight, cost)
                    pd.testing.assert_frame_equal(trades[["symbol", "entry_ts", "exit_ts"]],
                                                  book[["symbol", "entry_ts", "exit_ts"]])
                    if cost == 68 and weight == .2:
                        np.testing.assert_allclose(trades.notional, book.notional)
                        np.testing.assert_allclose(trades.net, book.net)
                    curve = portfolio_curve(trades, stocks, markets, START, END)
                    np.testing.assert_allclose(curve.iloc[-1], 1+(trades.notional*trades.net).sum())
                    returns, risk = daily_risk(curve)
                    usage, capital = capital_usage(trades, stocks, markets, curve)
                    assert risk["sharpe_days"] == 22 and capital["max_positions"] <= 2
                    row = dict(scope=scope, delay_minutes=delay, stock_cost_bps=cost,
                        weight_pct=int(round(weight*100)), start_utc=str(START), end_utc=str(END),
                        stock_trades=int(trades.asset.eq("stock").sum()),
                        crypto_trades=int(trades.asset.eq("crypto").sum()),
                        **metrics(trades, curve), **risk, **drawdown_durations(curve), **capital,
                        worst_full_day_pct=returns.min()*100,
                        stock_funding_contribution_pct=(trades.notional*trades.funding).sum()*100)
                    summaries.append(row)
                    tag = f"{scope}_{delay}_{cost}_{row['weight_pct']}"
                    trades.to_csv(OUT / f"trades_{tag}.csv", index=False)
                    usage.to_csv(OUT / f"capital_{tag}.csv.gz", compression="gzip")
                    returns.to_csv(OUT / f"daily_returns_{tag}.csv")
                    for asset, group in trades.groupby("asset"):
                        components.append(dict(scope=scope, delay_minutes=delay, stock_cost_bps=cost,
                            weight_pct=row["weight_pct"], asset=asset, trades=len(group),
                            wins=int(group.net.gt(0).sum()), contribution_pct=(group.notional*group.net).sum()*100))
        print(f"September shared/paired weights finished: stock delay={delay}m", flush=True)
    summary = pd.DataFrame(summaries)
    assert len(summary) == 60
    for _, group in summary.groupby(["scope", "delay_minutes", "stock_cost_bps"]):
        assert group.trades.nunique() == group.wins.nunique() == 1
    summary.to_csv(OUT / "summary.csv", index=False)
    pd.DataFrame(components).to_csv(OUT / "components.csv", index=False)

    actual = pd.read_csv(record(AUDIT / "closed_positions_enriched.csv"))
    for col in ("cTime", "uTime"):
        actual[col] = pd.to_datetime(actual[col], utc=True)
    actual = actual[actual.cTime.ge(START) & actual.uTime.lt(END)].copy()
    actual["win"] = actual.realizedPnl.gt(0)
    actual.groupby("asset").agg(trades=("realizedPnl", "size"), wins=("win", "sum"),
        gross_usdt=("pnl", "sum"), fees_usdt=("fee", "sum"), funding_usdt=("fundingFee", "sum"),
        net_usdt=("realizedPnl", "sum")).to_csv(OUT / "actual_closed_summary.csv")

    for path in (SHARPE_PROTOCOL, Path(__file__), Path("data/stocks_swap/universe.csv"),
                 Path("scripts/analysis/shared_two_slot_weights.py"),
                 Path("scripts/analysis/live_month_replay.py"),
                 Path("scripts/analysis/stock_portfolio_robustness.py"),
                 Path("scripts/analysis/stock_funding_robustness.py"),
                 Path("scripts/analysis/stock_robustness_study.py")):
        record(path)
    (OUT / "manifest.json").write_text(json.dumps(dict(input_and_code_hashes=hashes,
        start_utc=str(START), end_utc=str(END), weights=WEIGHTS, shared_slots=2,
        rows=len(summary), crypto_candidate_count=len(crypto),
        crypto_only_control_trades=len(control), actual_completed_positions=len(actual),
        sharpe="Full UTC calendar days, arithmetic returns, ddof=1, sqrt(365), risk-free=0; 22 days.",
        caveats=["Retrospective hybrid replay, NOT actual account performance, fills, deployment regimes or a fresh holdout.",
                 "Stocks reconstructed under corrected rules; crypto sides are first archived predictions, not newly trained outputs.",
                 "Archive logging gaps remain missing; no predictions fabricated. Later same-bar revisions not replayed.",
                 "Cash start; known deadlines must precede end; no pre-window inventory or terminal open positions.",
                 "All eligible crypto observations compete for two slots; no six-trade sleeve preselection.",
                 "Stock funding approximation included; crypto funding omitted, not assumed empirically zero.",
                 "Stock delays are scenarios. Crypto entry waits until next full observed 15m open.",
                 "Sizing defers fees/funding until close; costs use fixed round-trip bp, fractional lots and no margin rejection.",
                 "No historical mark-price margin, order loss, liquidation, size-dependent impact or actual account Sharpe.",
                 "Only 22 daily returns; sample annualized Sharpe is unstable and ignores autocorrelation corrections."]), indent=2))
    print(summary.query("scope == 'shared_live_crypto' and stock_cost_bps == 68")[
        ["delay_minutes", "weight_pct", "trades", "stock_trades", "crypto_trades", "return_pct",
         "marked_dd_pct", "sharpe_daily", "win_pct"]].to_string(index=False))


if __name__ == "__main__":
    main()
