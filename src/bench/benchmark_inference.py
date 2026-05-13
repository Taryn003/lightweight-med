from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch

from inference import InferenceConfig, QwenVLMReporter


def gpu_mem_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024**2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark model latency/throughput/memory")
    parser.add_argument("--image", required=True)
    parser.add_argument("--note", required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--out", default="output/benchmark.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    note = Path(args.note).read_text(encoding="utf-8").strip() if Path(args.note).exists() else args.note

    cfg = InferenceConfig(model_id=args.model_id)
    model = QwenVLMReporter(cfg)

    for _ in range(args.warmup):
        _ = model.generate_report(args.image, note)

    lat = []
    ok = 0
    fail = 0
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None

    for _ in range(args.runs):
        st = time.perf_counter()
        try:
            _ = model.generate_report(args.image, note)
            ok += 1
        except Exception:
            fail += 1
        lat.append(time.perf_counter() - st)

    total = sum(lat) if lat else 0.0
    result = {
        "model_id": args.model_id,
        "vlm_load_in_8bit": os.getenv("VLM_LOAD_IN_8BIT", "").lower() in {"1", "true", "yes", "on"},
        "runs": args.runs,
        "success": ok,
        "failed": fail,
        "p50_latency_s": sorted(lat)[len(lat) // 2] if lat else 0.0,
        "p95_latency_s": sorted(lat)[int(len(lat) * 0.95) - 1] if lat else 0.0,
        "throughput_rps": (ok / total) if total > 0 else 0.0,
        "gpu_peak_mem_mb": gpu_mem_mb(),
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
