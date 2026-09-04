"""Run the three long-lived data and trading processes in one container.

Docker normally gives each service its own container, but the deployment target
for this project intentionally uses one named container. This small supervisor
keeps the child processes isolated, prefixes their logs, restarts an unexpected
exit, and forwards shutdown signals cleanly.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
COMMANDS = {
    "stock-data": [sys.executable, "-m", "scripts.live.stock_data_loop"],
    "crypto-data": [sys.executable, "scripts/data/update_okx_crypto.py"],
    "demo": [sys.executable, "-m", "scripts.live.auto_demo"],
}


def stream(name: str, pipe) -> None:
    for line in iter(pipe.readline, ""):
        print(f"[{name}] {line.rstrip()}", flush=True)
    pipe.close()


def main() -> None:
    children: dict[str, subprocess.Popen[str]] = {}
    stopping = False

    def stop(_signum, _frame) -> None:
        nonlocal stopping
        stopping = True
        for child in children.values():
            if child.poll() is None:
                child.terminate()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    print("okx-quant supervisor starting: stock-data, crypto-data, demo", flush=True)
    try:
        while not stopping:
            for name, command in COMMANDS.items():
                child = children.get(name)
                if child is not None and child.poll() is None:
                    continue
                if child is not None:
                    print(f"[{name}] exited rc={child.returncode}; restarting", flush=True)
                    time.sleep(2)
                child = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                                         stderr=subprocess.STDOUT, text=True, bufsize=1,
                                         env=os.environ.copy())
                children[name] = child
                threading.Thread(target=stream, args=(name, child.stdout), daemon=True).start()
            time.sleep(1)
    finally:
        for child in children.values():
            if child.poll() is None:
                child.terminate()
        deadline = time.monotonic() + 20
        for child in children.values():
            remaining = max(0.1, deadline - time.monotonic())
            try:
                child.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                child.kill()
        print("okx-quant supervisor stopped", flush=True)


if __name__ == "__main__":
    main()
