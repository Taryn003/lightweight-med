from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build demo pairs for image+clinical note inference")
    parser.add_argument(
        "--metadata",
        default="data/iu_xray/metadata.jsonl",
        help="Input metadata jsonl from download_iu_xray.py",
    )
    parser.add_argument("--out", default="data/iu_xray/demo_pairs.jsonl", help="Output pairs jsonl")
    parser.add_argument("--max-samples", type=int, default=100, help="Maximum pairs to write")
    return parser.parse_args()


def first_sentence(text: str) -> str:
    clean = re.sub(r"\s+", " ", text).strip()
    for sep in [". ", ".", "。", "!", "?"]:
        if sep in clean:
            return clean.split(sep)[0].strip()
    return clean[:120].strip()


def make_pseudo_note(report: str) -> str:
    summary = first_sentence(report)
    return (
        "主诉：体检发现胸部影像异常，需进一步评估。\n"
        "现病史：无完整门诊病历，以下为历史影像摘要线索。\n"
        f"历史线索：{summary}\n"
        "请结合当前影像给出初步分析。"
    )


def main() -> None:
    args = parse_args()
    in_path = Path(args.metadata)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    wrote = 0
    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            if wrote >= args.max_samples:
                break
            row = json.loads(line)
            report = (row.get("report") or "").strip()
            image_path = row.get("image_path")
            if not report or not image_path:
                continue

            sample = {
                "id": row["id"],
                "image_path": image_path,
                "clinical_note": make_pseudo_note(report),
                "target_report": report,
                "note_type": "pseudo_from_report",
            }
            fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
            wrote += 1

    print(f"Wrote {wrote} demo pairs to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
