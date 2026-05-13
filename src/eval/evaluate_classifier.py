from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate auxiliary classifier predictions")
    parser.add_argument("--pred", required=True, help="jsonl with fields: label, prob")
    parser.add_argument("--out", default="output/classifier_metrics.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = [json.loads(x) for x in Path(args.pred).read_text(encoding="utf-8").splitlines() if x.strip()]
    y = np.array([int(r["label"]) for r in rows])
    p = np.array([float(r["prob"]) for r in rows])
    y_hat = (p >= 0.5).astype(int)

    result = {
        "n_samples": int(len(rows)),
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else 0.0,
        "f1": float(f1_score(y, y_hat, zero_division=0)),
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
