#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")" && pwd)
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python3}"
[ -x .venv/bin/python ] && PYTHON_BIN=".venv/bin/python"

usage() {
    cat <<'EOF'
用法:
  ./run.sh              增量更新数据并回测 b/c 的 tail 0.01/0.02 四格
  ./run.sh check        检查 OKX 模拟盘连接
  ./run.sh help         显示帮助
EOF
}

build_data() {
    "$PYTHON_BIN" -u -m scripts.data.fetch_binance --force
    "$PYTHON_BIN" -u -m scripts.data.build
}

run_backtest() {
    "$PYTHON_BIN" -u -m scripts.analysis.selftest
    # Daily candidates: 15m configurations b (1h) and c (2h),
    # compared at the two tails we are actually considering.
    for cfg in b c; do
        echo "=== train $cfg ==="
        "$PYTHON_BIN" -u -m scripts.modeling.train_dir "$cfg"
        echo "=== backtest $cfg ==="
        "$PYTHON_BIN" -u -m scripts.backtest.backtest_dir "$cfg" --tails=0.01,0.02
    done
    echo "=== report ==="
    "$PYTHON_BIN" -u -m scripts.backtest.report
}

case "${1:-all}" in
    all)
        build_data
        run_backtest
        ;;
    check)
        exec "$ROOT/scripts/run_quant.sh" --check
        ;;
    help|-h|--help)
        usage
        ;;
    *)
        echo "未知命令: $1" >&2
        usage >&2
        exit 2
        ;;
esac
