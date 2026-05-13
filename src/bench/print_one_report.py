"""单次推理并打印报告全文（用于 FP16 / INT8 文本对比）。依赖环境变量 VLM_LOAD_IN_8BIT。"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", required=True, help="本地目录或 HF id，覆盖 .env 中 MODEL_ID")
    p.add_argument("--image", required=True)
    p.add_argument("--note-file", default="", help="病历摘要文件路径")
    p.add_argument("--note", default="", help="病历摘要原文（与 --note-file 二选一）")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv()
    os.environ["MODEL_ID"] = args.model_id

    if args.note_file:
        note = Path(args.note_file).read_text(encoding="utf-8").strip()
    else:
        note = (args.note or "").strip()
    if not note:
        raise SystemExit("请提供 --note-file 或 --note")

    from inference import InferenceConfig, QwenVLMReporter

    img = args.image
    if not Path(img).is_file():
        raise SystemExit(f"影像不存在：{img}")

    cfg = InferenceConfig(model_id=args.model_id)
    rep = QwenVLMReporter(cfg)
    text = rep.generate_report(img, note)
    print(text)


if __name__ == "__main__":
    main()
