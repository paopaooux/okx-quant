"""Periodic safe-mode synchronizer for the OKX demo account.

This intentionally never passes ``--trade``.  Strategy execution will be a
separate, explicitly enabled component once signal and risk checks are wired.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time


def main() -> None:
    interval = max(15, int(os.environ.get("OKX_SYNC_INTERVAL", "60")))
    while True:
        result = subprocess.run([sys.executable, "-m", "scripts.live.okx_demo"], check=False)
        if result.returncode:
            print(f"sync failed (exit={result.returncode}); retrying in {interval}s", flush=True)
        time.sleep(interval)


if __name__ == "__main__":
    main()
