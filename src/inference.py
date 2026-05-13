from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from dotenv import load_dotenv
from transformers import AutoProcessor

from core.report_schema import SECTION_ORDER, StructuredReport, parse_structured_report
from core.vlm_loader import load_vlm_model

try:
    from qwen_vl_utils import process_vision_info
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "qwen-vl-utils is required. Install dependencies via `pip install -r requirements.txt`."
    ) from exc


def _strip_reasoning_wrappers(text: str) -> str:
    """去掉 Qwen3 等在正文外的 think 推理片段（尖括号包裹、与 tokenizer 特殊符号一致），否则后续无法按 `[章节]` 解析。"""
    t = text or ""
    # 與 tokenizer added_tokens 一致（151667/151668）；標籤名為 think（不是 thinking）。
    _OPEN, _CLOSE = "<think>", "</think>"
    patterns = (
        rf"{re.escape(_OPEN)}[\s\S]*?{re.escape(_CLOSE)}",
        r"<thinking>[\s\S]*?</thinking>",
        r"<redacted_reasoning>[\s\S]*?</redacted_reasoning>",
    )
    prev = None
    while prev != t:
        prev = t
        for p in patterns:
            t = re.sub(p, "", t, flags=re.IGNORECASE).strip()
    return t.strip()


DEFAULT_PROMPT = """
你是一名基层医疗影像辅助诊断AI。请基于输入的医学影像和病历文本，生成仅供医生参考的初步分析。
请严格使用以下结构输出，不要省略标题：

[初步结论]
- 给出1-3条结论，明确不确定性。

[影像关键征象]
- 列出可见征象，使用专业术语和简短解释。

[风险等级]
- 低/中/高，并说明依据。

[建议检查]
- 给出2-4条下一步检查建议。

[处理建议]
- 给出基层可执行建议，并明确“需医生最终判断”。

[患者科普]
- 用患者能理解的话，80字以内。

注意：
1) 不可给出绝对诊断结论。
2) 若影像质量不足，请明确指出。
3) 输出中文。
""".strip()


@dataclass
class InferenceConfig:
    model_id: str = "Qwen/Qwen3-VL-4B-Instruct"
    max_new_tokens: int = 384
    temperature: float = 0.2
    top_p: float = 0.9
    max_retries: int = 2
    retry_wait_seconds: float = 1.0
    do_sample: bool = True


def _inference_config_from_env(model_id: str) -> InferenceConfig:
    raw_nt = int(os.getenv("INFERENCE_MAX_NEW_TOKENS", "384"))
    max_new_tokens = max(128, min(raw_nt, 4096))
    deterministic = os.getenv("INFERENCE_DETERMINISTIC", "false").lower() in {"1", "true", "yes", "on"}
    if deterministic:
        do_sample = False
    else:
        do_sample = os.getenv("INFERENCE_DO_SAMPLE", "true").lower() in {"1", "true", "yes", "on"}
    return InferenceConfig(
        model_id=model_id,
        max_new_tokens=max_new_tokens,
        temperature=float(os.getenv("INFERENCE_TEMPERATURE", "0.2")),
        top_p=float(os.getenv("INFERENCE_TOP_P", "0.9")),
        max_retries=int(os.getenv("INFERENCE_MAX_RETRIES", "2")),
        retry_wait_seconds=float(os.getenv("INFERENCE_RETRY_WAIT_SECONDS", "1.0")),
        do_sample=do_sample,
    )


class QwenVLMReporter:
    def __init__(self, config: Optional[InferenceConfig] = None) -> None:
        load_dotenv()
        self.logger = logging.getLogger("qwen_vl_reporter")
        mid = os.getenv("MODEL_ID", "Qwen/Qwen3-VL-4B-Instruct")
        self.config = config or _inference_config_from_env(mid)

        self.model = load_vlm_model(
            self.config.model_id,
            dtype="auto",
        )
        self._device = next(self.model.parameters()).device
        self.logger.info(
            "vlm_loaded | param_device=%s | dtype=%s",
            self._device,
            next(self.model.parameters()).dtype,
        )
        self.processor = AutoProcessor.from_pretrained(self.config.model_id)

    def _build_messages(self, image_path: str, clinical_note: str) -> list[dict]:
        return [
            {"role": "system", "content": [{"type": "text", "text": DEFAULT_PROMPT}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {
                        "type": "text",
                        "text": (
                            "病历摘要如下，请结合影像进行分析：\n"
                            f"{clinical_note.strip()}\n\n"
                            "请输出结构化报告草稿。"
                        ),
                    },
                ],
            },
        ]

    def _generate_once(self, image_path: str, clinical_note: str) -> str:
        messages = self._build_messages(image_path, clinical_note)
        prompt_text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[prompt_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self._device)

        gen_kw: dict = {
            "max_new_tokens": self.config.max_new_tokens,
            "do_sample": self.config.do_sample,
        }
        if self.config.do_sample:
            gen_kw["temperature"] = self.config.temperature
            gen_kw["top_p"] = self.config.top_p
        else:
            seed_s = os.getenv("INFERENCE_SEED", "42")
            try:
                seed = int(seed_s)
            except ValueError:
                seed = 42
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        t0 = time.perf_counter()
        with torch.inference_mode():
            generated_ids = self.model.generate(**inputs, **gen_kw)

        trimmed_ids = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self.processor.batch_decode(
            trimmed_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        elapsed = time.perf_counter() - t0
        out_chars = len(response.strip())
        self.logger.info(
            "vlm_generate_done | %.2fs | max_new_tokens=%s | do_sample=%s | out_chars=%s",
            elapsed,
            self.config.max_new_tokens,
            self.config.do_sample,
            out_chars,
        )
        return response.strip()

    def generate_report(self, image_path: str, clinical_note: str) -> str:
        image_file = Path(image_path)
        if not image_file.exists():
            raise FileNotFoundError(f"Image not found: {image_file}")
        if not clinical_note.strip():
            raise ValueError("Clinical note is empty.")

        last_err: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            try:
                return self._generate_once(str(image_file), clinical_note)
            except Exception as exc:  # pragma: no cover
                last_err = exc
                self.logger.warning(
                    "inference attempt %s failed: %s", attempt + 1, exc, exc_info=False
                )
                if attempt < self.config.max_retries:
                    time.sleep(self.config.retry_wait_seconds)

        raise RuntimeError(f"inference failed after retries: {last_err}") from last_err

    def generate_structured(self, image_path: str, clinical_note: str) -> StructuredReport:
        raw = self.generate_report(image_path=image_path, clinical_note=clinical_note)
        cleaned = _strip_reasoning_wrappers(raw)
        structured = parse_structured_report(cleaned)
        if any((structured.sections.get(k) or "").strip() for k in SECTION_ORDER):
            return structured
        body = cleaned.strip() or raw.strip()
        sections = {k: "" for k in SECTION_ORDER}
        sections["初步结论"] = (
            body
            if body
            else "模型未返回可解析正文（可能只有推理标签或生成为空）。请增大 max_new_tokens，或确认模型能按提示输出六段 `[章节]` 结构。"
        )
        self.logger.warning(
            "structured_parse_empty_falling_back | raw_len=%s cleaned_len=%s",
            len(raw or ""),
            len(cleaned or ""),
        )
        return StructuredReport(raw_text=raw, sections=sections)
