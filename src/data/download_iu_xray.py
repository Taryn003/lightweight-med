from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download IU X-ray image-report pairs")
    parser.add_argument(
        "--repo-id",
        default="ayyuce/Indiana_University_Chest_X-ray_Collection",
        help="Hugging Face dataset repo id",
    )
    parser.add_argument("--split", default="train", help="Dataset split to use")
    parser.add_argument("--max-samples", type=int, default=200, help="Maximum number of samples")
    parser.add_argument("--out-dir", default="data/iu_xray", help="Output directory")
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Use streaming mode to avoid downloading full dataset upfront",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    image_dir = out_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: repo={args.repo_id}, split={args.split}, streaming={args.streaming}")
    ds = load_dataset(args.repo_id, split=args.split, streaming=args.streaming)

    metadata_path = out_dir / "metadata.jsonl"
    kept = 0
    with metadata_path.open("w", encoding="utf-8") as f:
        for i, row in enumerate(ds):
            if kept >= args.max_samples:
                break

            image = row.get("image")
            report = (row.get("report") or "").strip()
            question = (row.get("question") or "").strip()

            if image is None or not report:
                continue

            image_name = f"iu_{kept:05d}.png"
            image_path = image_dir / image_name
            image.save(image_path)
            # Store a portable relative path (relative to out_dir), so the jsonl
            # can be moved across machines without breaking absolute paths.
            rel_image_path = image_path.relative_to(out_dir)

            item = {
                "id": f"iu_{kept:05d}",
                "source_repo": args.repo_id,
                "source_split": args.split,
                "source_index": i,
                "image_path": str(rel_image_path),
                "question": question,
                "report": report,
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            kept += 1
            if kept % 10 == 0:
                print(f"Saved {kept} samples...")

    print(f"Saved {kept} samples to: {out_dir.resolve()}")
    print(f"Metadata: {metadata_path.resolve()}")


if __name__ == "__main__":
    main()
