#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
cd ..
PYTHON_BIN="${PYTHON_BIN:-python3}"
[ -x .venv/bin/python ] && PYTHON_BIN=".venv/bin/python"
exec "$PYTHON_BIN" -m scripts.live.okx_demo "$@"
