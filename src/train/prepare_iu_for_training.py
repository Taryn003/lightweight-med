from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

NORMAL_WORDS = {"normal"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare IU dataset for report generation + classification")
    parser.add_argument("--root", default="/Users/xiaoye/Downloads/archive", help="IU dataset root")
    parser.add_argument("--out-dir", default="data/iu_train", help="Output directory")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def split_terms(text: str) -> list[str]:
    if not isinstance(text, str):
        return []
    return [x.strip().lower() for x in text.split(";") if x.strip()]


def is_abnormal(row: pd.Series) -> int:
    for field in ("Problems", "MeSH"):
        terms = split_terms(str(row.get(field, "")))
        if terms and set(terms) == NORMAL_WORDS:
            return 0
    return 1


def build_clinical_note(row: pd.Series) -> str:
    indication = str(row.get("indication", "")).strip()
    comparison = str(row.get("comparison", "")).strip()
    findings = str(row.get("findings", "")).strip()
    if findings.lower() == "nan":
        findings = ""

    parts = ["主诉：胸部影像评估。"]
    if indication and indication.lower() != "nan":
        parts.append(f"检查指征：{indication}")
    if comparison and comparison.lower() != "nan":
        parts.append(f"对比信息：{comparison}")
    if findings:
        parts.append(f"历史线索：{findings[:220]}")
    parts.append("请结合当前影像给出结构化报告。")
    return "\n".join(parts)


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    reports = pd.read_csv(root / "indiana_reports.csv")
    proj = pd.read_csv(root / "indiana_projections.csv")

    frontal = proj[proj["projection"].str.lower() == "frontal"].drop_duplicates("uid")
    merged = reports.merge(frontal[["uid", "filename"]], on="uid", how="inner")

    img_root = root / "images" / "images_normalized"
    merged["image_path"] = merged["filename"].map(lambda x: str((img_root / x).resolve()))
    merged = merged[merged["image_path"].map(lambda p: Path(p).exists())].copy()

    merged["target_report"] = merged["impression"].fillna("").astype(str).str.strip()
    merged = merged[merged["target_report"] != ""]

    merged["clinical_note"] = merged.apply(build_clinical_note, axis=1)
    merged["label_abnormal"] = merged.apply(is_abnormal, axis=1)

    merged = merged.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    n = len(merged)
    n_train = int(n * args.train_ratio)
    n_val = int(n * args.val_ratio)

    splits = {
        "train": merged.iloc[:n_train],
        "val": merged.iloc[n_train : n_train + n_val],
        "test": merged.iloc[n_train + n_val :],
    }

    fields = [
        "uid",
        "image_path",
        "clinical_note",
        "target_report",
        "label_abnormal",
        "Problems",
        "MeSH",
    ]

    for name, df in splits.items():
        path = out_dir / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for _, row in df[fields].iterrows():
                f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")

    stat = {
        "total": n,
        "train": len(splits["train"]),
        "val": len(splits["val"]),
        "test": len(splits["test"]),
        "abnormal_ratio": float(merged["label_abnormal"].mean()),
    }
    (out_dir / "stats.json").write_text(json.dumps(stat, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stat, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
