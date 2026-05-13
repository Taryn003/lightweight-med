from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict

SECTION_ORDER = [
    "初步结论",
    "影像关键征象",
    "风险等级",
    "建议检查",
    "处理建议",
    "患者科普",
]


@dataclass
class StructuredReport:
    raw_text: str
    sections: Dict[str, str]

    @property
    def completeness(self) -> float:
        ok = sum(1 for k in SECTION_ORDER if self.sections.get(k, "").strip())
        return ok / len(SECTION_ORDER)


_SECTION_RE = re.compile(
    r"\[(初步结论|影像关键征象|风险等级|建议检查|处理建议|患者科普)\]\s*:?\s*([\s\S]*?)(?=\n\[(?:初步结论|影像关键征象|风险等级|建议检查|处理建议|患者科普)\]|\Z)",
    re.MULTILINE,
)


def _normalize_report_headers(text: str) -> str:
    """將模型常用寫法（全形括號、中式書名號標題）統一成解析器期待的 `[章节名]`。"""
    t = text or ""
    t = t.replace("\uff3b", "[").replace("\uff3d", "]")
    for name in SECTION_ORDER:
        t = re.sub(rf"【\s*{re.escape(name)}\s*】", f"[{name}]", t)
    return t


def parse_structured_report(text: str) -> StructuredReport:
    raw_orig = text or ""
    sections = {k: "" for k in SECTION_ORDER}
    for candidate in (raw_orig, _normalize_report_headers(raw_orig)):
        sections = {k: "" for k in SECTION_ORDER}
        for m in _SECTION_RE.finditer(candidate):
            name = m.group(1).strip()
            content = m.group(2).strip()
            sections[name] = content
        if any(v.strip() for v in sections.values()):
            break
    return StructuredReport(raw_text=raw_orig, sections=sections)
