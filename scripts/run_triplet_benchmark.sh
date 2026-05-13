#!/usr/bin/env bash
# 一次跑完：Qwen3-VL-4B vs Qwen3-VL-2B 底模 vs student_distilled_2b（延遲/顯存峰值）
# 用法：在 conda 環境中執行
#   bash scripts/run_triplet_benchmark.sh
# 快速試跑（每個模型只測 1 次）：
#   RUNS=1 bash scripts/run_triplet_benchmark.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

RUNS="${RUNS:-3}"
IMAGE="${IMAGE:-$ROOT/data/iu_xray/images/iu_00005.png}"
NOTE="${NOTE:-$ROOT/examples/sample_note.txt}"
OUT="${OUT:-$ROOT/output/benchmark_triplet.json}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/mnt/data_3/yjt/qwen_model/hf_cache}"

python "$ROOT/src/bench/benchmark_variants.py" \
  --config "$ROOT/configs/benchmark_variants.yaml" \
  --project-root "$ROOT" \
  --image "$IMAGE" \
  --note "$NOTE" \
  --runs "$RUNS" \
  --out "$OUT"

echo ""
echo "汇总 JSON: $OUT"
echo "分项: $ROOT/output/benchmark_qwen3_vl_4b_instruct.json 等"
