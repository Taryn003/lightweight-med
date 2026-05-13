from __future__ import annotations

import argparse
from pathlib import Path

from inference import QwenVLMReporter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a multimodal medical draft report")
    parser.add_argument("--image", required=True, help="Path to medical image")
    parser.add_argument("--note", default="", help="Short clinical note text")
    parser.add_argument("--note-file", default="", help="Optional path to a text note file")
    parser.add_argument(
        "--out",
        default="output/report.md",
        help="Output markdown path (default: output/report.md)",
    )
    return parser.parse_args()


def load_note(note: str, note_file: str) -> str:
    if note_file:
        return Path(note_file).read_text(encoding="utf-8").strip()
    return note.strip()


def main() -> None:
    args = parse_args()
    clinical_note = load_note(args.note, args.note_file)
    if not clinical_note:
        raise ValueError("Please pass --note or --note-file")

    reporter = QwenVLMReporter()
    report = reporter.generate_report(args.image, clinical_note)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report + "\n", encoding="utf-8")

    print("=== 结构化报告草稿 ===")
    print(report)
    print(f"\nSaved to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
