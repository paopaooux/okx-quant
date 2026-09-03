"""Safe OKX demo helper for tokenized-stock instruments.

This is intentionally a small execution adapter.  It never places an order
unless ``--trade`` is supplied.  Strategy signal generation remains in the
backtest runner until a live SEC polling policy is explicitly enabled.
"""

from __future__ import annotations

import argparse

from scripts.live.okx_demo import DemoClient


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="查询模拟盘余额和代币化股票现货标的")
    parser.add_argument("--trade", action="store_true",
                        help="显式提交一笔模拟盘现货市价单")
    parser.add_argument("--inst-id", help="例如 XMSTR-USDT")
    parser.add_argument("--side", choices=("buy", "sell"))
    parser.add_argument("--sz", help="下单数量，按标的最小单位")
    args = parser.parse_args()
    if not args.check and not args.trade:
        parser.error("请指定 --check 或 --trade；默认不会执行任何操作")

    client = DemoClient()
    if args.check:
        balance = client.balance()
        rows = client._request("GET", "/api/v5/public/instruments",
                               params={"instType": "SPOT"})
        stocks = [row for row in rows if row.get("instCategory") == "3"
                  and row.get("quoteCcy") == "USDT"]
        print(f"模拟盘账户: {balance[0].get('totalEq') if balance else 'unknown'}")
        print(f"代币化股票现货标的: {len(stocks)}")
        for row in stocks[:20]:
            print(f"  {row.get('instId')} state={row.get('state')} "
                  f"minSz={row.get('minSz')} lever={row.get('lever')}")

    if args.trade:
        if not all((args.inst_id, args.side, args.sz)):
            parser.error("--trade 必须同时提供 --inst-id、--side、--sz")
        rows = client._request("GET", "/api/v5/public/instruments",
                               params={"instType": "SPOT"})
        allowed = {
            row.get("instId") for row in rows
            if row.get("instCategory") == "3" and row.get("quoteCcy") == "USDT"
        }
        if args.inst_id not in allowed:
            parser.error("--inst-id 必须是 OKX USDT 代币化股票（instCategory=3）")
        result = client.order(args.inst_id, args.side, args.sz, td_mode="cash")
        print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
