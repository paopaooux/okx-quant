"""Frozen combination execution rules, without data feeds or account clients."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import pandas as pd


@dataclass(frozen=True)
class CombinationPolicy:
    version: str = "original_combo_v1"
    stock_pool: str = "all"
    stock_trigger_bps: float = 600.
    stock_stop_bps: float = 300.
    stock_take_bps: float = 0.
    stock_resolve_minutes: int = 60
    stock_max_hours: int = 30
    stock_slots: int = 3
    stock_entries_per_utc_day: int = 2
    shared_slots: int = 5
    slot_weight: float = .2
    crypto_config: str = "c"
    crypto_tail: float = .01
    crypto_horizon_bars: int = 48

    def manifest(self):
        return asdict(self)


POLICY = CombinationPolicy()
BAR_MS = 15 * 60 * 1000


def stock_instruments(universe):
    return sorted(universe.instId.dropna().astype(str).unique())


def stock_deadline(event_ts, resolve_ts):
    event = pd.Timestamp(event_ts)
    resolve = pd.Timestamp(resolve_ts)
    if pd.isna(event) or pd.isna(resolve):
        raise ValueError("Stock entries require event and cash-open timestamps")
    return min(resolve + pd.Timedelta(minutes=POLICY.stock_resolve_minutes),
               event + pd.Timedelta(hours=POLICY.stock_max_hours))


def entry_metadata(signal):
    stock = signal["asset_type"] == "stock"
    deadline = (stock_deadline(signal["event_ts"], signal["resolve_ts"]) if stock else
                pd.to_datetime(signal["bar"] + (POLICY.crypto_horizon_bars + 1) * BAR_MS, unit="ms", utc=True))
    return dict(policy_version=POLICY.version, take_width=0. if stock else signal["width"],
                deadline_ts=deadline.isoformat(), resolve_ts=signal.get("resolve_ts"),
                event_ts=signal.get("event_ts"), opened_bar=signal["bar"])


def migrate_policy_state(state, now):
    """Do not reinterpret trades opened before this policy was deployed."""
    first = state.get("policy_version") != POLICY.version
    if first:
        state["policy_version"] = POLICY.version
        state["policy_activated_at"] = now.isoformat()
        # The old engine did not retain a reliable daily-entry counter. Do not
        # give an already-trading account two additional entries on migration.
        used = POLICY.stock_entries_per_utc_day if state.get("trades") or state.get("positions") else 0
        state.setdefault("stock_entries_by_day", {}).setdefault(str(now.date()), used)
    state.setdefault("pending_entries", {})
    for pos in state["positions"].values():
        if pos.get("policy_version"):
            continue
        pos["policy_version"] = "legacy_preserved"
        pos.setdefault("take_width", float(pos.get("width") or 0.))
        opened = int(pos.get("opened_bar") or 0)
        if opened and not pos.get("deadline_ts"):
            duration = pd.Timedelta(hours=30 if pos.get("asset_type") == "stock" else 12)
            pos["deadline_ts"] = (pd.to_datetime(opened, unit="ms", utc=True) + duration).isoformat()
    return state


def entry_rejection(state, signal, now):
    if signal.get("management_only") or signal.get("side") not in {"long", "short"}:
        return "management_only_or_flat"
    active = {p["inst_id"]: p for p in state["positions"].values()}
    active.update({p["inst_id"]: p for p in state.get("pending_entries", {}).values()})
    if signal["inst_id"] in active:
        return "same_instrument"
    if len(active) >= POLICY.shared_slots:
        return "shared_capacity"
    if signal["asset_type"] == "stock":
        if sum(p.get("asset_type") == "stock" for p in active.values()) >= POLICY.stock_slots:
            return "stock_capacity"
        used = state.get("stock_entries_by_day", {}).get(str(now.date()), 0)
        if used >= POLICY.stock_entries_per_utc_day:
            return "stock_daily_limit"
    return None


def exit_reason(pos, mark, now):
    entry = float(pos.get("entry_px") or 0.)
    width = float(pos.get("width") or 0.)
    take = float(pos.get("take_width", width))
    if entry > 0 and math.isfinite(mark) and mark > 0:
        move = (mark / entry - 1) * (1 if pos["side"] == "long" else -1)
        if width > 0 and move <= -width + 1e-12:
            return "stop"
        if take > 0 and move >= take - 1e-12:
            return "target"
    if pos.get("deadline_ts") and now >= pd.Timestamp(pos["deadline_ts"]):
        return "deadline"
    return None
