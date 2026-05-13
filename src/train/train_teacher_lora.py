from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import AutoProcessor, Trainer, TrainingArguments

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
    max_pixels: int | None = None
    assistant_only_loss: bool = True

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        messages = []
        prompt_messages = []
        for row in batch:
            image_item = {"type": "image", "image": row["image_path"]}
            if self.max_pixels is not None:
                image_item["max_pixels"] = self.max_pixels
            msg = [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {
                    "role": "user",
                    "content": [
                        image_item,
                        {"type": "text", "text": row["clinical_note"]},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": row["target_report"]}],
                },
            ]
            messages.append(msg)
            prompt_messages.append(msg[:-1])

        texts = [self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=False) for m in messages]
        prompt_texts = [
            self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in prompt_messages
        ]
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
        if self.assistant_only_loss:
            prompt_inputs = self.processor(
                text=prompt_texts,
                padding=True,
                return_tensors="pt",
            )
            prompt_lens = prompt_inputs["attention_mask"].sum(dim=1).tolist()
            for i, plen in enumerate(prompt_lens):
                labels[i, : int(plen)] = -100
        features["labels"] = labels
        return features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA finetune teacher model on IU report generation")
    parser.add_argument("--train", default="data/iu_train/train.jsonl")
    parser.add_argument("--val", default="data/iu_train/val.jsonl")
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--out", default="output/teacher_lora")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--max-pixels", type=int, default=512 * 512)
    parser.add_argument(
        "--assistant-only-loss",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_ds = IUReportDataset(args.train)
    val_ds = IUReportDataset(args.val)

    processor = AutoProcessor.from_pretrained(args.model_id)
    base = load_vlm_model(args.model_id, dtype="auto", device_map="auto")

    peft_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base, peft_cfg)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    training_args = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_steps=100,
        save_total_limit=2,
        bf16=torch.cuda.is_available(),
        report_to=[],
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=Collator(
            processor=processor,
            max_pixels=args.max_pixels,
            assistant_only_loss=args.assistant_only_loss,
        ),
    )
    trainer.train()
    trainer.save_model(args.out)
    processor.save_pretrained(args.out)
    print(f"saved teacher lora to: {args.out}")


if __name__ == "__main__":
    main()
