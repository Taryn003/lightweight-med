#!/usr/bin/env bash
set -euo pipefail

export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HOME=${HF_HOME:-/mnt/data_3/yjt/qwen_model/hf_cache}

echo "[1/4] prepare dataset"
python src/train/prepare_iu_for_training.py --root /Users/xiaoye/Downloads/archive --out-dir data/iu_train

echo "[2/4] train teacher lora"
python src/train/run_with_config.py --script src/train/train_teacher_lora.py --config configs/train_teacher_lora.yaml

echo "[3/4] distill student"
python src/train/run_with_config.py --script src/train/distill_student.py --config configs/distill_student.yaml

echo "[4/4] benchmark"
python src/bench/benchmark_variants.py \
  --config configs/benchmark_variants.yaml \
  --project-root "$(pwd)" \
  --image data/iu_xray/images/iu_00005.png \
  --note examples/sample_note.txt \
  --runs 2
