#!/usr/bin/env bash
# Full re-run on the extended 2021-2026 sample.
# Expanding window only first -- the rolling variants were rejected on the 3-year
# sample and re-running all nine panels before knowing whether the base result
# survives would just re-inflate the multiple-testing multiplicity.
set -e
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-python3}"
[ -x .venv/bin/python ] && PYTHON_BIN=".venv/bin/python"
"$PYTHON_BIN" -u -m scripts.analysis.selftest
for c in b c; do
  echo "=== train $c ==="; "$PYTHON_BIN" -u -m scripts.modeling.train_dir "$c"
  echo "=== backtest $c (tail 0.01/0.02) ===";
  "$PYTHON_BIN" -u -m scripts.backtest.backtest_dir "$c" --tails=0.01,0.02
done
echo "=== report ==="; "$PYTHON_BIN" -u -m scripts.backtest.report
