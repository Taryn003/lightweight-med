from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run train/distill script from yaml config")
    parser.add_argument("--script", required=True, help="python script path")
    parser.add_argument("--config", required=True, help="yaml config path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    cmd = ["python", args.script]
    for k, v in cfg.items():
        flag = f"--{k.replace('_', '-') }"
        cmd.extend([flag, str(v)])

    print("running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
