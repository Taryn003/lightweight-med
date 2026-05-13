from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run benchmark across model variants")
    parser.add_argument("--config", default="configs/benchmark_variants.yaml")
    parser.add_argument(
        "--project-root",
        default="",
        help="專案根目錄（含 src/、configs/）。默認取 configs 的上一級目錄。",
    )
    parser.add_argument("--image", required=True)
    parser.add_argument("--note", required=True)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--out", default="output/benchmark_variants.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config).expanduser().resolve()
    repo_root = Path(args.project_root).expanduser().resolve() if args.project_root else cfg_path.parent.parent

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    variants = cfg.get("variants", [])

    bench_script = repo_root / "src" / "bench" / "benchmark_inference.py"
    env = os.environ.copy()
    src_path = str(repo_root / "src")
    env["PYTHONPATH"] = src_path + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    all_results = []
    for item in variants:
        name = item["name"]
        model_id = item["model_id"]
        per_out = repo_root / "output" / f"benchmark_{name}.json"
        cmd = [
            sys.executable,
            str(bench_script),
            "--image",
            args.image,
            "--note",
            args.note,
            "--model-id",
            model_id,
            "--runs",
            str(args.runs),
            "--out",
            str(per_out),
        ]
        run_env = env.copy()
        for key, val in (item.get("env") or {}).items():
            run_env[str(key)] = str(val)

        print("running:", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True, cwd=repo_root, env=run_env)
        result = json.loads(per_out.read_text(encoding="utf-8"))
        result["name"] = name
        all_results.append(result)

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = repo_root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(all_results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
