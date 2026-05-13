#!/usr/bin/env bash
# 同一输入下先后打印 FP16 与 INT8 完整报告文本（需已 pip install bitsandbytes）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 2>/dev/null || command -v python)}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "请设置 PYTHON_BIN 或安装 python3" >&2
  exit 1
fi
MODEL_ID="${MODEL_ID:-output/student_distilled_2b}"
IMAGE="${IMAGE:-$ROOT/data/iu_xray/images/iu_00005.png}"
NOTE_PATH="${NOTE:-$ROOT/examples/sample_note.txt}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export MODEL_ID

run_one () {
  local label="$1"
  local q="$2"
  echo ""
  echo "======== ${label} ========"
  VLM_LOAD_IN_8BIT="$q" "$PYTHON_BIN" "$ROOT/src/bench/print_one_report.py" \
    --model-id "$MODEL_ID" \
    --image "$IMAGE" \
    --note-file "$NOTE_PATH"
}

run_one "student FP16 (VLM_LOAD_IN_8BIT=false)" "false"
run_one "student INT8 (VLM_LOAD_IN_8BIT=true)" "true"
