from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class ModelSpec:
    model_id: str
    display_name: str
    system_prompt: str
    torch_dtype: str
    max_new_tokens: int
    section_order: list[str]
    hf_cache_dir: str | None = None


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
- 给出基层可执行建议，并明确"需医生最终判断"。

[患者科普]
- 用患者能理解的话，80字以内。

注意：
1) 不可给出绝对诊断结论。
2) 若影像质量不足，请明确指出。
3) 输出中文。
""".strip()

DEFAULT_SECTIONS = [
    "初步结论",
    "影像关键征象",
    "风险等级",
    "建议检查",
    "处理建议",
    "患者科普",
]


def _build_registry() -> dict[str, ModelSpec]:
    return {
        "default": ModelSpec(
            model_id=os.getenv("MODEL_ID", "Qwen/Qwen2.5-VL-2B-Instruct"),
            display_name="Qwen2.5-VL-2B",
            system_prompt=DEFAULT_PROMPT,
            torch_dtype="float16",
            max_new_tokens=384,
            section_order=DEFAULT_SECTIONS,
            hf_cache_dir=None,
        ),
    }


MODEL_REGISTRY: dict[str, ModelSpec] = _build_registry()


def get_spec(key: str = "default") -> ModelSpec:
    if key not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model key: {key!r}. Available: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[key]
