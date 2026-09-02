"""Minimal OKX demo-trading client.

This module deliberately does not place orders unless ``--trade`` is supplied.
Credentials are read from OKX_API_KEY/OKX_API_SECRET/OKX_API_PASSPHRASE.
"""
from __future__ import annotations

import argparse, base64, hashlib, hmac, json, os, sqlite3, time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
import requests

BASE = "https://www.okx.com"
DB_PATH = Path(os.environ.get("OKX_DEMO_DB", Path(__file__).resolve().parents[2] / "data" / "okx_demo.sqlite3"))


def load_dotenv() -> None:
    """Load the local .env for the CLI without overwriting shell variables."""
    path = Path(__file__).resolve().parents[2] / ".env"
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if value[:1] == value[-1:] and value[:1] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


load_dotenv()
DB_PATH = Path(os.environ.get("OKX_DEMO_DB", DB_PATH))


def db_connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row
    db.executescript("""
    CREATE TABLE IF NOT EXISTS balance_snapshots (
      id INTEGER PRIMARY KEY, ts TEXT NOT NULL, total_eq TEXT, raw_json TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS orders (
      id INTEGER PRIMARY KEY, ts TEXT NOT NULL, inst_id TEXT NOT NULL,
      side TEXT NOT NULL, sz TEXT NOT NULL, td_mode TEXT NOT NULL,
      reduce_only INTEGER NOT NULL, ord_id TEXT, state TEXT, fill_sz TEXT,
      avg_px TEXT, fee TEXT, pnl TEXT, raw_json TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS position_snapshots (
      id INTEGER PRIMARY KEY, ts TEXT NOT NULL, inst_id TEXT NOT NULL,
      pos TEXT, pos_side TEXT, avg_px TEXT, mark_px TEXT, upl TEXT,
      upl_ratio TEXT, liq_px TEXT, margin TEXT, lever TEXT, raw_json TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS fills (
      trade_id TEXT PRIMARY KEY, ts TEXT NOT NULL, inst_id TEXT NOT NULL,
      ord_id TEXT, side TEXT, fill_px TEXT, fill_sz TEXT, fee TEXT,
      fee_ccy TEXT, exec_type TEXT, raw_json TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS bills (
      bill_id TEXT PRIMARY KEY, ts TEXT NOT NULL, bill_type TEXT,
      sub_type TEXT, inst_id TEXT, currency TEXT, amount TEXT,
      balance TEXT, raw_json TEXT NOT NULL
    );
    """)
    # Keep databases created by earlier demo versions usable.
    columns = {row[1] for row in db.execute("PRAGMA table_info(balance_snapshots)")}
    if "total_eq" not in columns:
        db.execute("ALTER TABLE balance_snapshots ADD COLUMN total_eq TEXT")
    return db


def save_balance(rows):
    total_eq = rows[0].get("totalEq") if rows and isinstance(rows[0], dict) else None
    with db_connect() as db:
        db.execute("INSERT INTO balance_snapshots(ts, total_eq, raw_json) VALUES (?, ?, ?)",
                   (datetime.now(timezone.utc).isoformat(), total_eq, json.dumps(rows, ensure_ascii=True)))


def save_account_data(positions, fills, bills):
    now = datetime.now(timezone.utc).isoformat()
    with db_connect() as db:
        for p in positions:
            db.execute("""INSERT INTO position_snapshots
              (ts,inst_id,pos,pos_side,avg_px,mark_px,upl,upl_ratio,liq_px,margin,lever,raw_json)
              VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""", (now, p.get("instId"), p.get("pos"), p.get("posSide"),
              p.get("avgPx"), p.get("markPx"), p.get("upl"), p.get("uplRatio"), p.get("liqPx"),
              p.get("margin"), p.get("lever"), json.dumps(p, ensure_ascii=True)))
        for f in fills:
            tid = f.get("tradeId") or f.get("fillId")
            if tid:
                db.execute("""INSERT OR IGNORE INTO fills
                  (trade_id,ts,inst_id,ord_id,side,fill_px,fill_sz,fee,fee_ccy,exec_type,raw_json)
                  VALUES (?,?,?,?,?,?,?,?,?,?,?)""", (tid, now, f.get("instId"), f.get("ordId"), f.get("side"),
                  f.get("fillPx"), f.get("fillSz"), f.get("fee"), f.get("feeCcy"), f.get("execType"),
                  json.dumps(f, ensure_ascii=True)))
        for b in bills:
            bid = b.get("billId")
            if bid:
                db.execute("""INSERT OR IGNORE INTO bills
                  (bill_id,ts,bill_type,sub_type,inst_id,currency,amount,balance,raw_json)
                  VALUES (?,?,?,?,?,?,?,?,?)""", (bid, now, b.get("type"), b.get("subType"), b.get("instId"),
                  b.get("ccy"), b.get("balChg"), b.get("bal"), json.dumps(b, ensure_ascii=True)))


def print_stats():
    with db_connect() as db:
        row = db.execute("""SELECT COUNT(*) orders,
            COALESCE(SUM(CAST(pnl AS REAL)), 0) pnl,
            COALESCE(SUM(ABS(CAST(fee AS REAL))), 0) fees,
            COALESCE(SUM(CAST(fill_sz AS REAL)), 0) volume
            FROM orders WHERE state IN ('filled','partially_filled')""").fetchone()
        fill_row = db.execute("""SELECT COUNT(*) count,
            COALESCE(SUM(CAST(fill_sz AS REAL)),0) volume,
            COALESCE(SUM(ABS(CAST(fee AS REAL))),0) fees FROM fills""").fetchone()
        bill_row = db.execute("""SELECT COALESCE(SUM(CASE WHEN sub_type IN ('173','funding_fee') THEN CAST(amount AS REAL) ELSE 0 END),0) funding,
            COALESCE(SUM(CASE WHEN sub_type IN ('5','fee') THEN CAST(amount AS REAL) ELSE 0 END),0) fees FROM bills""").fetchone()
        equity = db.execute("SELECT total_eq FROM balance_snapshots WHERE total_eq IS NOT NULL ORDER BY id").fetchall()
        positions = db.execute("""SELECT inst_id,
            SUM(CASE WHEN side='buy' THEN CAST(fill_sz AS REAL) ELSE -CAST(fill_sz AS REAL) END) net_sz
            FROM orders WHERE state IN ('filled','partially_filled') GROUP BY inst_id
            HAVING ABS(net_sz) > 1e-12 ORDER BY inst_id""").fetchall()
        print(f"database: {DB_PATH}")
        print(f"orders: {row['orders']}  filled volume: {row['volume']:.8g}")
        print(f"realized pnl: {row['pnl']:.8f} USDT  fees: {row['fees']:.8f} USDT")
        print(f"fills: {fill_row['count']}  fill volume: {fill_row['volume']:.8g}  funding: {bill_row['funding']:.8f}  bill fees: {bill_row['fees']:.8f}")
        if len(equity) >= 2:
            initial, latest = float(equity[0][0]), float(equity[-1][0])
            roi = (latest / initial - 1.0) * 100 if initial else 0.0
            print(f"equity: {latest:.8f} USDT  return: {roi:.6f}% (since first snapshot)")
        print("net positions:")
        for p in positions:
            print(f"  {p['inst_id']}: {p['net_sz']:.8g} contracts")

class DemoClient:
    def __init__(self):
        self.key = os.environ.get("OKX_API_KEY")
        self.secret = os.environ.get("OKX_API_SECRET")
        self.passphrase = os.environ.get("OKX_API_PASSPHRASE")
        if not all((self.key, self.secret, self.passphrase)):
            raise RuntimeError("请设置 OKX_API_KEY、OKX_API_SECRET、OKX_API_PASSPHRASE")
        self.s = requests.Session()
        # OKX's edge may reject a custom/programmatic UA before the request
        # reaches the API.  Keep this client usable in the same environments as
        # the repository's data fetcher.
        self.s.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/126.0.0.0 Safari/537.36",
            "Accept": "application/json",
            "Content-Type": "application/json",
        })
        proxy = os.environ.get("OKX_PROXY_URL")
        if proxy:
            self.s.proxies.update({"http": proxy, "https": proxy})

    def _request(self, method, path, body=None, params=None):
        body_text = json.dumps(body, separators=(",", ":")) if body else ""
        query = urlencode(params or {}, doseq=True)
        request_path = path + ("?" + query if query else "")
        ts = datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
        msg = ts + method.upper() + request_path + body_text
        sign = base64.b64encode(hmac.new(self.secret.encode(), msg.encode(), hashlib.sha256).digest()).decode()
        headers = {"OK-ACCESS-KEY": self.key, "OK-ACCESS-SIGN": sign,
                   "OK-ACCESS-TIMESTAMP": ts, "OK-ACCESS-PASSPHRASE": self.passphrase,
                   "x-simulated-trading": "1"}
        # Pass the already-encoded query in the URL so it is byte-for-byte the
        # same path that was signed above.
        r = self.s.request(method, BASE + request_path, data=body_text, headers=headers, timeout=15)
        try:
            p = r.json()
        except ValueError:
            r.raise_for_status()
            raise RuntimeError(f"OKX returned non-JSON HTTP {r.status_code}")
        if r.status_code >= 400 or p.get("code") != "0":
            raise RuntimeError({"http_status": r.status_code,
                                "okx_code": p.get("code"),
                                "okx_msg": p.get("msg"),
                                "data": p.get("data")})
        return p.get("data", [])

    def balance(self):
        return self._request("GET", "/api/v5/account/balance")

    def order_detail(self, inst_id, ord_id):
        for attempt in range(5):
            rows = self._request("GET", "/api/v5/trade/order", params={"instId": inst_id, "ordId": ord_id})
            detail = rows[0] if rows else {}
            if detail.get("state") in {"filled", "canceled", "mmp_canceled", "partially_filled"}:
                return detail
            if attempt < 4:
                time.sleep(0.3)
        return detail

    def positions(self):
        return self._request("GET", "/api/v5/account/positions", params={"instType": "SWAP"})

    def fills(self):
        return self._request("GET", "/api/v5/trade/fills", params={"instType": "SWAP", "limit": "100"})

    def bills(self):
        return self._request("GET", "/api/v5/account/bills", params={"instType": "SWAP", "limit": "100"})

    def instruments(self):
        return self._request("GET", "/api/v5/public/instruments", params={"instType": "SWAP"})

    def order(self, inst_id, side, sz, td_mode="isolated", reduce_only=False):
        body = {"instId": inst_id, "tdMode": td_mode, "side": side, "ordType": "market", "sz": str(sz)}
        if reduce_only: body["reduceOnly"] = "true"
        return self._request("POST", "/api/v5/trade/order", body=body)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="check demo balance and instrument metadata")
    ap.add_argument("--trade", action="store_true", help="place one demo market order")
    ap.add_argument("--inst-id", help="OKX instrument, e.g. BTC-USDT-SWAP")
    ap.add_argument("--side", choices=("buy", "sell"), help="buy opens long / sell opens short in net mode")
    ap.add_argument("--sz", help="order size in contracts (not coins)")
    ap.add_argument("--td-mode", choices=("cross", "isolated"), default="isolated")
    ap.add_argument("--reduce-only", action="store_true", help="close/reduce a net-mode position")
    ap.add_argument("--stats", action="store_true", help="show locally recorded order/PnL statistics")
    args = ap.parse_args()
    if args.stats:
        print_stats()
        return
    if args.trade:
        missing = [name for name, value in (("--inst-id", args.inst_id), ("--side", args.side), ("--sz", args.sz)) if not value]
        if missing:
            ap.error("--trade requires " + ", ".join(missing))
        print("WARNING: --trade will place OKX DEMO orders only (x-simulated-trading=1).")
    try:
        c = DemoClient()
    except RuntimeError as exc:
        ap.error(str(exc) + "；可复制 .env.example 为 .env 后填写")
    try:
        balance = c.balance()
    except (requests.RequestException, RuntimeError, ValueError) as exc:
        ap.error(f"OKX 连接/认证失败: {exc}")
    save_balance(balance)
    print("demo balance:", balance)
    try:
        positions, fills, bills = c.positions(), c.fills(), c.bills()
        save_account_data(positions, fills, bills)
        print(f"synced: {len(positions)} positions, {len(fills)} fills, {len(bills)} bills")
    except (requests.RequestException, RuntimeError, ValueError) as exc:
        print(f"account data sync warning: {exc}")
    if args.check:
        rows = c.instruments()
        for x in rows:
            if x.get("instId") in {"BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP"}:
                print(x)
    if args.trade:
        try:
            result = c.order(args.inst_id, args.side, args.sz, args.td_mode, args.reduce_only)
        except (requests.RequestException, RuntimeError, ValueError) as exc:
            ap.error(f"OKX 下单失败: {exc}")
        ord_id = result[0].get("ordId") if result else None
        detail = c.order_detail(args.inst_id, ord_id) if ord_id else {}
        row = {
            "ts": datetime.now(timezone.utc).isoformat(), "inst_id": args.inst_id,
            "side": args.side, "sz": str(args.sz), "td_mode": args.td_mode,
            "reduce_only": int(args.reduce_only), "ord_id": ord_id,
            "state": detail.get("state", "submitted"), "fill_sz": detail.get("accFillSz", "0"),
            "avg_px": detail.get("avgPx", "0"), "fee": detail.get("fee", "0"),
            "pnl": detail.get("pnl", "0"), "raw_json": json.dumps({"submit": result, "detail": detail}, ensure_ascii=True),
        }
        with db_connect() as db:
            db.execute("""INSERT INTO orders(ts,inst_id,side,sz,td_mode,reduce_only,ord_id,state,
                         fill_sz,avg_px,fee,pnl,raw_json) VALUES
                         (:ts,:inst_id,:side,:sz,:td_mode,:reduce_only,:ord_id,:state,
                          :fill_sz,:avg_px,:fee,:pnl,:raw_json)""", row)
        save_balance(c.balance())
        print("demo order:", result)
        print("filled detail:", detail)
    else:
        print("safe mode: no orders sent; signal engine is not connected yet")

if __name__ == "__main__": main()
