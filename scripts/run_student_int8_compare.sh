#!/usr/bin/env bash
# 蒸餾 2B：FP16 vs INT8（bitsandbytes）性能对比，汇总 output/benchmark_student_quant.json
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 2>/dev/null || command -v python)}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "请设置 PYTHON_BIN 或安装 python3：例如 export PYTHON_BIN=/path/to/conda/env/bin/python" >&2
  exit 1
fi
RUNS="${RUNS:-3}"
IMAGE="${IMAGE:-$ROOT/data/iu_xray/images/iu_00005.png}"
NOTE="${NOTE:-$ROOT/examples/sample_note.txt}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/mnt/data_3/yjt/qwen_model/hf_cache}"

"$PYTHON_BIN" "$ROOT/src/bench/benchmark_variants.py" \
  --config "$ROOT/configs/benchmark_student_int8.yaml" \
  --project-root "$ROOT" \
  --image "$IMAGE" \
  --note "$NOTE" \
  --runs "$RUNS" \
  --out "$ROOT/output/benchmark_student_quant.json"

echo "汇总: $ROOT/output/benchmark_student_quant.json"
