from __future__ import annotations

import argparse
import json
from pathlib import Path

from inference import QwenVLMReporter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multimodal inference on demo pairs")
    parser.add_argument("--pairs", default="data/iu_xray/demo_pairs.jsonl", help="Input pairs jsonl")
    parser.add_argument("--out", default="output/demo_predictions.jsonl", help="Output jsonl path")
    parser.add_argument("--max-samples", type=int, default=5, help="How many samples to run")
    return parser.parse_args()


def resolve_image_path(raw_path: str, pairs_path: Path) -> Path:
    p = Path(raw_path)
    if p.is_absolute() and p.exists():
        return p

    # Preferred: resolve relative paths from pairs file location.
    if not p.is_absolute():
        candidate = (pairs_path.parent / p).resolve()
        if candidate.exists():
            return candidate

        # Fallback: resolve relative to current working directory.
        candidate = p.resolve()
        if candidate.exists():
            return candidate

    # Backward compatibility: old jsonl may contain absolute paths from another machine.
    # Try matching by file name under the local pairs/images directory.
    fallback = (pairs_path.parent / "images" / p.name).resolve()
    if fallback.exists():
        return fallback

    raise FileNotFoundError(f"Image not found: {raw_path}")


def main() -> None:
    args = parse_args()
    pairs_path = Path(args.pairs)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    reporter = QwenVLMReporter()

    wrote = 0
    with pairs_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            if wrote >= args.max_samples:
                break
            row = json.loads(line)
            image_path = resolve_image_path(row["image_path"], pairs_path)
            clinical_note = row["clinical_note"]

            structured = reporter.generate_structured(image_path=str(image_path), clinical_note=clinical_note)
            record = {
                "id": row["id"],
                "image_path": str(image_path),
                "clinical_note": clinical_note,
                "target_report": row.get("target_report", ""),
                "prediction": structured.raw_text,
                "sections": structured.sections,
                "structured_completeness": structured.completeness,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            wrote += 1
            print(f"[{wrote}] done: {row['id']}")

    print(f"Saved {wrote} predictions to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
