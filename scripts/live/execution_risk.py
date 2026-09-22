"""Durable order recovery and exchange-side protection for the single writer."""
from __future__ import annotations

import time
import uuid
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR

from scripts.live.okx_demo import OKXAPIError


TERMINAL = {"filled", "canceled", "mmp_canceled"}
ALGO_TERMINAL = {"effective", "canceled", "order_failed", "partially_effective"}


def defer_retry(record, prefix="retry", error=None):
    attempts = int(record.get(prefix + "_attempts", 0)) + 1
    record[prefix + "_attempts"] = attempts
    record[prefix + "_after"] = time.time() + min(300, 5 * 2 ** min(attempts - 1, 6))
    if error is not None:
        record[prefix + "_error"] = str(error)


def release_entry(state, client_id, *, retry=False, error=None):
    pending = state["pending_entries"].pop(client_id)
    day = pending.get("quota_day")
    if day:
        ledger = state.setdefault("stock_entries_by_day", {})
        ledger[day] = max(0, ledger.get(day, 0) - 1)
    if retry:
        key = pending["inst_id"] + "|" + pending["side"]
        record = state.setdefault("entry_retries", {}).setdefault(key, {})
        defer_retry(record, error=error)
        record["bar"] = pending.get("opened_bar", 0)
        # Permit the same still-fresh signal to be retried after backoff.
        state.setdefault("last_bars", {}).pop(key, None)


def definitely_absent(client, intent, client_id, error):
    """Only retire a lost regular-order submission after its gateway expiry.

    Never infer absence from a network error or a single early lookup miss.
    Old intents without a known expiry continue to reserve their slots.
    """
    if not isinstance(error, OKXAPIError) or error.code != "51603" or intent.get("ord_id"):
        return False
    expires = int(intent.get("exp_time") or 0)
    if not expires or time.time() * 1000 < expires + 60_000:
        return False
    orders = client.pending_orders(intent["inst_id"])
    if any(o.get("clOrdId") == client_id for o in orders):
        return False
    positions = client.positions()
    if any(p.get("instId") == intent["inst_id"] and float(p.get("pos") or 0) != 0 for p in positions):
        return False
    intent["absence_checks"] = int(intent.get("absence_checks", 0)) + 1
    return intent["absence_checks"] >= 3


def recover_missing_exit(client, pos, error):
    if not isinstance(error, OKXAPIError) or error.code != "51603" or pos.get("exit_ord_id"):
        return False
    expires = int(pos.get("exit_exp_time") or 0)
    if not expires or time.time() * 1000 < expires + 60_000:
        return False
    if any(o.get("clOrdId") == pos["exit_client_id"] for o in client.pending_orders(pos["inst_id"])):
        return False
    remote = [p for p in client.positions() if p.get("instId") == pos["inst_id"]
              and (float(p.get("pos") or 0) > 0 if pos["side"] == "long" else float(p.get("pos") or 0) < 0)
              and p.get("mgnMode", "isolated") == "isolated"]
    if len(remote) != 1:
        return False
    pos["exit_absence_checks"] = int(pos.get("exit_absence_checks", 0)) + 1
    if pos["exit_absence_checks"] < 3:
        return False
    # Expired ingress plus repeated negative lookups permits another strictly
    # reduce-only attempt, capped at the current position rather than old size.
    pos["size"] = str(abs(float(remote[0]["pos"])))
    pos.pop("exit_client_id", None)
    pos.pop("exit_absence_checks", None)
    defer_retry(pos, "exit_retry", error)
    return True


def stop_price(pos, spec):
    entry, width = Decimal(str(pos["entry_px"])), Decimal(str(pos["width"]))
    tick = Decimal(str(spec["tickSz"]))
    if not all(x.is_finite() and x > 0 for x in (entry, width, tick)) or width >= 1:
        raise ValueError("invalid stop inputs")
    long = pos["side"] == "long"
    raw = entry * (1 - width if long else 1 + width)
    price = (raw / tick).to_integral_value(rounding=ROUND_CEILING if long else ROUND_FLOOR) * tick
    return format(price, "f")


def protect_positions(client, state, specs, save):
    """One full-position reduce-only stop, persisted before submission."""
    registry = state.setdefault("protective_orders", {})
    for key, pos in state["positions"].items():
        if pos.get("exit_client_id") or pos.get("force_exit"):
            continue
        if float(pos.get("width") or 0) <= 0:
            # Legacy crypto positions without a recoverable stop cannot safely
            # accept more exposure; their existing deadline still applies.
            state["entry_reconciliation_ok"] = False
            print(f"PROTECTION MISSING WIDTH {pos['inst_id']}", flush=True)
            continue
        cid = pos.get("stop_client_id")
        if cid:
            record = registry[cid]
            if time.time() < record.get("check_after", 0):
                continue
            try:
                detail = client.algo_detail(cid, record.get("algo_id"))
                record["algo_id"] = detail.get("algoId") or record.get("algo_id")
                if detail.get("state") == "live":
                    record["check_after"] = time.time() + 15
                    record["confirmed"] = True
                elif detail.get("state") in ALGO_TERMINAL:
                    # The remote position may lag a triggered stop. A reduce-only
                    # exit is safe; never install a second stop on that snapshot.
                    pos["force_exit"] = "protection_terminal"
                    registry.pop(cid)
                    pos.pop("stop_client_id", None)
                else:
                    raise RuntimeError("unconfirmed protective order state")
            except Exception as exc:
                state["entry_reconciliation_ok"] = False
                print(f"PROTECTION UNKNOWN {pos['inst_id']}: {type(exc).__name__}", flush=True)
            save(state)
            continue
        if any(r["inst_id"] == pos["inst_id"] for r in registry.values()):
            state["entry_reconciliation_ok"] = False
            continue
        try:
            trigger = stop_price(pos, specs[pos["inst_id"]])
        except (KeyError, ValueError, ArithmeticError):
            pos["force_exit"] = "invalid_protection_metadata"
            state["entry_reconciliation_ok"] = False
            save(state)
            continue
        cid = "sl" + uuid.uuid4().hex[:30]
        pos["stop_client_id"] = cid
        registry[cid] = {"inst_id": pos["inst_id"], "position_key": key, "trigger": trigger}
        save(state)
        try:
            rows = client.stop_order(pos["inst_id"], "sell" if pos["side"] == "long" else "buy", trigger, cid)
            if not rows or not rows[0].get("algoId"):
                raise RuntimeError("protective order acknowledgement missing")
            registry[cid]["algo_id"] = rows[0]["algoId"]
            detail = client.algo_detail(cid, rows[0]["algoId"])
            if detail.get("state") != "live":
                raise RuntimeError("protective order not live")
            registry[cid].update(confirmed=True, check_after=time.time() + 15)
        except Exception as exc:
            if isinstance(exc, OKXAPIError) and exc.rejected and not registry[cid].get("algo_id"):
                registry.pop(cid)
                pos.pop("stop_client_id", None)
            pos["force_exit"] = "protection_unavailable"
            state["entry_reconciliation_ok"] = False
            print(f"PROTECTION FAILED {pos['inst_id']}: {type(exc).__name__}", flush=True)
        save(state)


def cleanup_protection(client, state, save):
    registry = state.setdefault("protective_orders", {})
    active_ids = {p.get("stop_client_id") for p in state["positions"].values()}
    for cid, record in list(registry.items()):
        if cid in active_ids:
            continue
        try:
            detail = client.algo_detail(cid, record.get("algo_id"))
            if detail.get("state") in ALGO_TERMINAL:
                registry.pop(cid)
            elif detail.get("state") == "live" and detail.get("algoId"):
                record["algo_id"] = detail["algoId"]
                client.cancel_algo(record["inst_id"], detail["algoId"])
                # Retain the registry until a later read confirms cancellation.
            else:
                raise RuntimeError("unknown orphan stop state")
        except Exception as exc:
            state["entry_reconciliation_ok"] = False
            print(f"PROTECTION CLEANUP UNKNOWN {record['inst_id']}: {type(exc).__name__}", flush=True)
        save(state)
