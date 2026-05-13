from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor

from core.vlm_loader import load_vlm_model
from qwen_vl_utils import process_vision_info

SYSTEM_PROMPT = (
    "你是一名基层医疗影像辅助诊断AI。基于输入影像和病历文本，"
    "输出结构化报告，包含[初步结论][影像关键征象][风险等级][建议检查][处理建议][患者科普]。"
)


class IUReportDataset(Dataset):
    def __init__(self, path: str) -> None:
        self.rows = [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        return self.rows[idx]


@dataclass
class Collator:
    processor: AutoProcessor

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        messages = []
        for row in batch:
            msg = [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": row["image_path"]},
                        {"type": "text", "text": row["clinical_note"]},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": row["target_report"]}],
                },
            ]
            messages.append(msg)

        texts = [self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=False) for m in messages]
        image_inputs, video_inputs = process_vision_info(messages)
        features = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        labels = features["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        features["labels"] = labels
        return features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distill student from teacher")
    parser.add_argument("--train", default="data/iu_train/train.jsonl")
    parser.add_argument("--teacher", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--student", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--out", default="output/student_distilled")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--alpha-kd", type=float, default=0.7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = AutoProcessor.from_pretrained(args.student)
    ds = IUReportDataset(args.train)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=Collator(processor))

    teacher = load_vlm_model(args.teacher, dtype="auto", device_map="auto")
    student = load_vlm_model(args.student, dtype="auto", device_map="auto")

    teacher.eval()
    student.train()

    opt = torch.optim.AdamW(student.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        for step, batch in enumerate(loader, start=1):
            batch = {k: v.to(student.device) for k, v in batch.items()}
            labels = batch["labels"]

            with torch.no_grad():
                t_out = teacher(**batch)

            s_out = student(**batch)
            task_loss = s_out.loss

            mask = labels != -100
            t_logits = t_out.logits[mask] / args.temperature
            s_logits = s_out.logits[mask] / args.temperature
            kd_loss = F.kl_div(
                F.log_softmax(s_logits, dim=-1),
                F.softmax(t_logits, dim=-1),
                reduction="batchmean",
            ) * (args.temperature**2)

            loss = args.alpha_kd * kd_loss + (1.0 - args.alpha_kd) * task_loss
            opt.zero_grad()
            loss.backward()
            opt.step()

            if step % 20 == 0:
                print(
                    json.dumps(
                        {
                            "epoch": epoch + 1,
                            "step": step,
                            "loss": float(loss.item()),
                            "task_loss": float(task_loss.item()),
                            "kd_loss": float(kd_loss.item()),
                        },
                        ensure_ascii=False,
                    )
                )

    Path(args.out).mkdir(parents=True, exist_ok=True)
    student.save_pretrained(args.out)
    processor.save_pretrained(args.out)
    print(f"saved distilled student to: {args.out}")


if __name__ == "__main__":
    main()
