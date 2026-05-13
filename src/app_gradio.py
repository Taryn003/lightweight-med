from __future__ import annotations

import json
import logging
import os
import re
import shutil
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import gradio as gr
import pandas as pd

from core.case_store import append_case, delete_case, get_case, list_cases_filtered
from core.report_schema import SECTION_ORDER
from inference import QwenVLMReporter

reporter: QwenVLMReporter | None = None
LOG_FILE = Path("output/app_events.log")
APP_CSS = """
/* 主题色：改 --brand / --brand-dark 即可对接院方色板 */
:root {
  --brand: #e97830;
  --brand-dark: #c85f1f;
  --bg-soft: #fff7f0;
  --card: #ffffff;
  --line: #f1d4c0;
  --text: #2a2a2a;
  --muted: #6f6f6f;
}
.gradio-container {
  background:
    radial-gradient(circle at 8% 4%, rgba(233,120,48,0.10), transparent 26%),
    radial-gradient(circle at 94% 2%, rgba(255,193,158,0.22), transparent 30%),
    linear-gradient(180deg, #fffaf7 0%, #fff 26%);
  color: var(--text);
}
.hero {
  border: 1px solid var(--line);
  background: linear-gradient(135deg, #fff8f3 0%, #fff 60%);
  border-radius: 16px;
  padding: 16px 18px;
  margin-bottom: 10px;
}
.hero-with-art {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 20px;
  flex-wrap: wrap;
}
.hero-text { flex: 1 1 220px; min-width: 0; }
.hero-art {
  flex: 0 0 auto;
  opacity: 0.95;
}
.hero-art svg {
  width: min(148px, 32vw);
  height: auto;
  display: block;
}
@media (max-width: 560px) {
  .hero-art { width: 100%; text-align: center; }
}
.hero-title {
  font-size: 28px;
  font-weight: 800;
  letter-spacing: 0.5px;
  margin-bottom: 6px;
}
.hero-sub {
  color: var(--muted);
  font-size: 15px;
}
.tag-row { margin-top: 10px; display: flex; gap: 8px; flex-wrap: wrap; }
.tag {
  font-size: 12px; padding: 5px 10px; border-radius: 999px;
  border: 1px solid #f0c9ad; color: #9a4b1b; background: #fff3ea;
}
.panel {
  background: var(--card);
  border: 1px solid #eee5de;
  border-radius: 14px;
  padding: 8px;
}
.panel-title {
  font-weight: 700;
  color: #5b5b5b;
  margin: 2px 0 8px 2px;
}
button.primary {
  background: linear-gradient(90deg, var(--brand) 0%, #ff974b 100%) !important;
  border: 0 !important;
}
button.primary:hover {
  background: linear-gradient(90deg, var(--brand-dark) 0%, #ee8640 100%) !important;
}
.result-box textarea {
  font-size: 14px !important;
  line-height: 1.65 !important;
}
.hint-card {
  font-size: 13px;
  line-height: 1.55;
  color: var(--muted);
  background: linear-gradient(165deg, #fffbf8 0%, #fff 55%);
  border: 1px solid var(--line);
  border-left: 4px solid var(--brand);
  border-radius: 12px;
  padding: 12px 14px;
  margin-bottom: 10px;
}
.hint-card strong { color: var(--text); font-weight: 700; }
.hint-card ul { margin: 8px 0 0 0; padding-left: 1.1rem; }
.hint-card li { margin-bottom: 4px; }
.note-toolbar-wrap { margin-bottom: 6px; }
.note-toolbar-wrap button { flex: 0 0 auto; }
.workbench-head {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 12px;
  flex-wrap: wrap;
  margin-bottom: 6px;
}
.workbench-head .wb-title {
  font-size: 15px;
  font-weight: 700;
  color: #5b5b5b;
}
.wb-title-inline {
  display: inline-flex;
  align-items: center;
  gap: 8px;
}
.wb-title-inline svg {
  flex-shrink: 0;
  opacity: 0.85;
}
.workbench-head .wb-steps {
  font-size: 13px;
  color: var(--muted);
}
.note-column textarea {
  min-height: 240px !important;
}
.disclaimer-bar {
  font-size: 12px;
  color: #8a7a72;
  margin: 6px 2px 12px 2px;
  padding: 8px 12px;
  background: rgba(255,255,255,0.75);
  border: 1px solid #eee5de;
  border-radius: 10px;
}
.readiness-box {
  font-size: 13px;
  padding: 10px 14px;
  border-radius: 12px;
  margin-bottom: 10px;
  border: 1px solid #eee5de;
}
.readiness-box.ok {
  color: #1a6b45;
  background: linear-gradient(165deg, #f3fcf8 0%, #fff 55%);
  border-color: #c5e8d8;
}
.readiness-box.warn {
  color: #8a4b15;
  background: linear-gradient(165deg, #fffbf5 0%, #fff 55%);
  border-color: var(--line);
}
/* 病例列表：限制预览列高度，便于扫读 */
.gradio-dataframe tbody td {
  max-height: 3.4em !important;
  overflow: hidden !important;
  vertical-align: top !important;
}
/* 生成状态：更像工作台提示条 */
.status-strip label {
  font-weight: 600 !important;
  color: #4a4a4a !important;
}
.status-strip textarea,
.status-strip input {
  border-left: 3px solid var(--brand) !important;
  background: linear-gradient(180deg, #fffdfb 0%, #fff 100%) !important;
}
/* 与 Gradio 组件标题对齐的「图标 + 文案」字段标签 */
.field-label-with-icon {
  display: flex;
  align-items: center;
  gap: 8px;
  margin: 8px 2px 6px 2px;
}
.field-label-with-icon .fld-ico svg {
  display: block;
  opacity: 0.88;
}
.field-label-with-icon .fld-text {
  font-size: 14px;
  font-weight: 600;
  color: #4a4a4a;
}
.field-label-with-icon .fld-sub {
  font-weight: 500;
  color: var(--muted);
  font-size: 13px;
  margin-left: 4px;
}
"""

CLINICAL_NOTE_PLACEHOLDER = (
    "主诉 / 现病史 / 既往与过敏 / 体征（体温·血压·脉氧） / 其他检查（可选）"
)

CLINICAL_NOTE_TEMPLATE = """主诉：咳嗽伴胸闷约3天。
现病史：起病逐渐加重，咳白色黏痰，活动后气促；近24小时无发热，食欲睡眠尚可。
既往史：高血压长期服药控制；青霉素过敏史。
体征：T 36.8℃，BP 128/80 mmHg，SpO2 96%（室内空气）。
辅助检查：暂无近期化验与外院正式报告；外院曾提示肺纹理增粗（请结合本次影像综合判断）。"""

CLINICAL_NOTE_HINT_HTML = """
<div class="hint-card">
  <strong>病历摘要要点</strong>
  <ul style="margin-top:6px;">
    <li><strong>主诉</strong> 症状+时间 · <strong>现病史</strong> 起病与变化 · <strong>既往/过敏</strong></li>
    <li><strong>体征</strong> T·BP·SpO₂ · <strong>其它</strong> 外院关键词即可（不可替代面诊）</li>
  </ul>
</div>
"""

HERO_BLOCK_HTML = """
<div class="hero hero-with-art">
  <div class="hero-text">
    <div class="hero-title">基层慧眼</div>
    <div class="hero-sub">轻量级多模态基层医疗影像智能辅助诊断系统</div>
    <div class="tag-row">
      <span class="tag">医数未来小组</span>
    </div>
  </div>
  <div class="hero-art" aria-hidden="true">
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 128 96" role="img">
      <defs>
        <linearGradient id="heroFilm" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" style="stop-color:#e97830;stop-opacity:0.28"/>
          <stop offset="100%" style="stop-color:#ffc49a;stop-opacity:0.12"/>
        </linearGradient>
      </defs>
      <rect x="10" y="14" width="108" height="72" rx="12" fill="url(#heroFilm)" stroke="#e4976a" stroke-width="1.3" opacity="0.95"/>
      <path d="M42 30 Q30 50 36 74 Q44 80 52 74 Q56 52 48 30 Z" fill="none" stroke="#c85f1f" stroke-width="1.5" stroke-linecap="round" opacity="0.75"/>
      <path d="M86 30 Q98 50 92 74 Q84 80 76 74 Q72 52 80 30 Z" fill="none" stroke="#c85f1f" stroke-width="1.5" stroke-linecap="round" opacity="0.75"/>
      <path d="M56 28 L72 28 L68 70 L60 70 Z" fill="#ffffff" fill-opacity="0.55" stroke="#d97340" stroke-width="1" opacity="0.85"/>
      <circle cx="28" cy="24" r="2.5" fill="#e97830" opacity="0.5"/>
      <circle cx="100" cy="22" r="2" fill="#e97830" opacity="0.35"/>
    </svg>
  </div>
</div>
"""

WORKBENCH_TITLE_HTML = """
<div class="workbench-head panel-title">
  <span class="wb-title wb-title-inline">
    <svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path d="M12 3v4M8 7h8M9 21h6v-6a3 3 0 0 0-6 0v6Z" stroke="#e97830" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"/>
      <path d="M7 11h10" stroke="#e97830" stroke-width="1.7" stroke-linecap="round" opacity="0.65"/>
    </svg>
    医生诊断工作台
  </span>
  <span class="wb-steps">影像 → 摘要 → 生成 → 保存</span>
</div>
"""

AI_DRAFT_LABEL_HTML = """
<div class="field-label-with-icon">
  <span class="fld-ico" aria-hidden="true">
    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none">
      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8l-6-6Z" stroke="var(--brand)" stroke-width="1.65" stroke-linecap="round" stroke-linejoin="round"/>
      <path d="M14 2v6h6M8 13h8M8 17h6M8 9h3" stroke="var(--brand)" stroke-width="1.5" stroke-linecap="round" opacity="0.82"/>
    </svg>
  </span>
  <span class="fld-text">AI 草稿<span class="fld-sub">（可改）</span></span>
</div>
"""


READINESS_MIN_CHARS = 12


def _can_generate(image_path: str | None, clinical_note: str) -> bool:
    img_ok = bool(image_path and str(image_path).strip())
    note_ok = len((clinical_note or "").strip()) >= READINESS_MIN_CHARS
    return img_ok and note_ok


def init_logging() -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"), logging.StreamHandler()],
    )


def log_event(event: str, payload: dict[str, Any]) -> None:
    logging.info("%s | %s", event, json.dumps(payload, ensure_ascii=False))


def get_reporter() -> QwenVLMReporter:
    global reporter
    if reporter is None:
        log = logging.getLogger("qwen_vl_reporter")
        t0 = time.perf_counter()
        reporter = QwenVLMReporter()
        log.info("gradio_vlm_singleton_ready | %.2fs", time.perf_counter() - t0)
    return reporter


def clinical_note_fill_template_ready(image_path: str | None) -> tuple[str, str, Any]:
    t = CLINICAL_NOTE_TEMPLATE
    html, btn_upd = readiness_ui(image_path, t)
    return t, html, btn_upd


def clinical_note_clear_ready(image_path: str | None) -> tuple[str, str, Any]:
    html, btn_upd = readiness_ui(image_path, "")
    return "", html, btn_upd


def readiness_ui(image_path: str | None, clinical_note: str) -> tuple[str, Any]:
    img_ok = bool(image_path and str(image_path).strip())
    note = (clinical_note or "").strip()
    nlen = len(note)
    note_ok = nlen >= READINESS_MIN_CHARS
    if img_ok and note_ok:
        inner = "✓ 可生成"
        cls = "readiness-box ok"
    else:
        missing = []
        if not img_ok:
            missing.append("上传影像")
        if not note_ok:
            missing.append(f"摘要≥{READINESS_MIN_CHARS}字（现{nlen}）")
        inner = "请先：" + " · ".join(missing)
        cls = "readiness-box warn"
    html = f'<div class="{cls}">{inner}</div>'
    return html, gr.update(interactive=_can_generate(image_path, clinical_note))


def _has_substantial_draft(text: str) -> bool:
    s = (text or "").strip()
    if len(s) < 36:
        return False
    return ("[初步结论]" in s) or ("[影像关键征象]" in s)


def infer(
    image_path: str,
    clinical_note: str,
    current_draft: str,
    confirm_overwrite: bool,
) -> tuple[Any, Any, str, Any]:
    try:
        if not image_path:
            return gr.update(), gr.update(), "缺少影像。", gr.update()
        if not clinical_note.strip():
            return gr.update(), gr.update(), "缺少病历摘要。", gr.update()

        substantial = _has_substantial_draft(current_draft)
        if substantial and not confirm_overwrite:
            return (
                gr.update(),
                gr.update(),
                "已有草稿：请勾选「覆盖已有草稿」后再生成。",
                gr.update(),
            )

        model = get_reporter()
        structured = model.generate_structured(image_path=image_path, clinical_note=clinical_note)
        cleaned = _clean_sections(structured.sections)
        pretty = "\n\n".join(
            f"[{k}]\n{cleaned.get(k, '').strip() or '(空)'}" for k in SECTION_ORDER
        )
        completeness = _calc_completeness(cleaned)
        n_sec = len(SECTION_ORDER)
        done = int(round(completeness * n_sec))
        if completeness >= 1.0:
            status = "已生成，六段齐全，请校对后保存。"
        else:
            status = f"已生成 {done}/{n_sec} 段，空段可在框内补。"
        log_event(
            "infer_success",
            {
                "image_path": image_path,
                "completeness": completeness,
                "clinical_note_len": len(clinical_note),
                "had_prior_draft": substantial,
            },
        )
        return pretty, pretty, status, gr.update(value=False)
    except Exception as exc:  # pragma: no cover
        logging.exception("infer_failed")
        return gr.update(), gr.update(), f"失败：{exc}", gr.update()


def _dedup_sentences(text: str, max_sentences: int = 6) -> str:
    parts = re.split(r"[。！？；;\n]+", text)
    seen = set()
    out = []
    for p in parts:
        s = p.strip(" -\t\r\n")
        if not s:
            continue
        key = re.sub(r"\s+", "", s)
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
        if len(out) >= max_sentences:
            break
    return "；".join(out) + ("。" if out else "")


def _clean_sections(sections: dict[str, str]) -> dict[str, str]:
    cleaned: dict[str, str] = {}
    for k in SECTION_ORDER:
        v = (sections.get(k) or "").strip()
        v = _dedup_sentences(v, max_sentences=8 if k == "初步结论" else 5)
        cleaned[k] = v
    return cleaned


def _calc_completeness(sections: dict[str, str]) -> float:
    ok = sum(1 for k in SECTION_ORDER if (sections.get(k) or "").strip())
    return ok / len(SECTION_ORDER)


CASE_IMAGES_DIR = Path("output/case_images")


def _safe_text(val: Any) -> str:
    if val is None:
        return ""
    if isinstance(val, float) and pd.isna(val):
        return ""
    return str(val)


def _format_saved_at(raw: str) -> str:
    s = (raw or "").strip()
    if not s:
        return ""
    try:
        if "T" in s:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
            return dt.strftime("%Y-%m-%d %H:%M")
    except ValueError:
        pass
    return s[:16]


def _short_image_label(name: str) -> str:
    if not name or name == "—":
        return name
    if len(name) <= 28:
        return name
    p = Path(name)
    stem, suf = p.stem, p.suffix
    if len(stem) <= 12:
        return name
    return f"{stem[:6]}…{stem[-3:]}{suf}"


def _persist_case_image(src_path: str, case_id: str) -> str:
    """将上传/临时影像复制到 output/case_images，避免 /tmp/gradio 清理后路径失效。"""
    src = Path(src_path)
    if not src.is_file():
        return src_path
    try:
        CASE_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        suf = src.suffix if src.suffix else ".png"
        dest = CASE_IMAGES_DIR / f"{case_id}{suf}"
        shutil.copy2(src, dest)
        return str(dest.resolve())
    except OSError as exc:
        logging.warning("persist_case_image_failed | %s", exc)
        return src_path


def save_report(
    image_path: str,
    clinical_note: str,
    ai_draft: str,
    edited_report: str,
) -> tuple[str, pd.DataFrame, list[str]]:
    df, ids = load_case_table()
    if not image_path:
        return "无影像路径。", df, ids
    if not edited_report.strip():
        return "报告为空。", df, ids

    case_id = str(uuid.uuid4())
    stable_image_path = _persist_case_image(image_path, case_id)

    payload = append_case(
        {
            "case_id": case_id,
            "image_path": stable_image_path,
            "clinical_note": clinical_note.strip(),
            "ai_draft": ai_draft.strip(),
            "doctor_edited_report": edited_report.strip(),
        }
    )
    log_event("save_case", {"case_id": payload["case_id"], "image_path": stable_image_path})
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    msg = f"已保存 · {ts} · {payload['case_id'][:8]}…"
    df2, ids2 = load_case_table()
    return msg, df2, ids2


CASE_TABLE_LIMIT = 500


def load_case_table() -> tuple[pd.DataFrame, list[str]]:
    rows = list_cases_filtered(limit=CASE_TABLE_LIMIT)
    cols = ["序号", "病例ID", "保存时间", "病历预览", "影像文件"]
    if not rows:
        return pd.DataFrame(columns=cols), []
    out = []
    ids: list[str] = []
    for i, r in enumerate(rows):
        cid = _safe_text(r.get("case_id"))
        ids.append(cid)
        raw_note = _safe_text(r.get("clinical_note"))
        preview = raw_note.replace("\n", " ").strip()
        if len(preview) > 52:
            preview = preview[:52] + "…"
        img = _safe_text(r.get("image_path"))
        img_name = Path(img).name if img else ""
        cid_short = (f"{cid[:8]}…") if len(cid) > 9 else (cid or "—")
        out.append(
            {
                "序号": str(i + 1),
                "病例ID": cid_short,
                "保存时间": _format_saved_at(_safe_text(r.get("saved_at"))),
                "病历预览": preview or "—",
                "影像文件": _short_image_label(img_name) if img_name else "—",
            }
        )
    return pd.DataFrame(out), ids


def delete_case_ui(
    selected_case_id: str,
    delete_confirmed: bool,
) -> tuple[pd.DataFrame, list[str], Any, Any, Any, Any, Any, str]:
    sid = (selected_case_id or "").strip()

    def pack(
        msg: str,
        *,
        clear_selection: bool = False,
        clear_loaded: bool = False,
    ):
        df, ids = load_case_table()
        sel_out: Any = "" if clear_selection else gr.update()
        chk = gr.update(value=False)
        if clear_loaded:
            return df, ids, chk, sel_out, "", "", "", msg
        return df, ids, chk, sel_out, gr.update(), gr.update(), gr.update(), msg

    if not sid:
        return pack("请先在表中点选一行。")
    if not delete_confirmed:
        return pack("删除不可恢复，请先勾选确认。")

    if not delete_case(sid):
        return pack("删除失败。")

    log_event("delete_case", {"case_id": sid})
    return pack(
        "已删除。",
        clear_selection=True,
        clear_loaded=True,
    )


def _select_row_index(evt: gr.SelectData) -> int | None:
    idx = evt.index
    if idx is None:
        return None
    try:
        if isinstance(idx, (list, tuple)):
            row = int(idx[0])
        else:
            row = int(idx)
    except (TypeError, ValueError):
        return None
    return row


def open_case(evt: gr.SelectData, case_ids: list[str]) -> tuple[str, str, str, str, str]:
    if not case_ids:
        return "", "", "", "无数据。", ""
    row_idx = _select_row_index(evt)
    if row_idx is None:
        return "", "", "", "请点击表格。", ""
    if row_idx < 0 or row_idx >= len(case_ids):
        return "", "", "", "无效行。", ""

    cid = case_ids[row_idx]
    case = get_case(cid)
    if not case:
        return "", "", "", "记录不存在。", ""

    note = _safe_text(case.get("clinical_note"))
    ai_d = _safe_text(case.get("ai_draft"))
    doc = _safe_text(case.get("doctor_edited_report"))

    img_path = _safe_text(case.get("image_path"))
    img_tip = ""
    if img_path and not Path(img_path).is_file():
        img_tip = " · 影像缺失"

    status = f"已加载 {cid[:8]}…" if len(cid) >= 8 else f"已加载 {cid or '?'}"
    if img_tip:
        status = f"{status}{img_tip}"
    return note, ai_d, doc, status, cid


init_logging()
_initial_df, _initial_case_ids = load_case_table()
_init_readiness_html, _ = readiness_ui(None, "")

with gr.Blocks(title="基层慧眼 - 多模态影像辅助诊断系统") as demo:
    gr.HTML(HERO_BLOCK_HTML)

    with gr.Tabs():
        with gr.Tab("工作台"):
            gr.HTML(WORKBENCH_TITLE_HTML)
            gr.HTML(
                '<div class="disclaimer-bar">AI 辅助草稿，临床请以您的判断为准。</div>'
            )
            with gr.Row():
                with gr.Column(elem_classes=["panel"]):
                    image = gr.Image(type="filepath", label="X光影像")
                with gr.Column(elem_classes=["panel", "note-column"]):
                    gr.HTML(CLINICAL_NOTE_HINT_HTML)
                    with gr.Row(elem_classes=["note-toolbar-wrap"]):
                        btn_note_template = gr.Button("示例模板", size="sm")
                        btn_note_clear = gr.Button("清空", size="sm")
                    note = gr.Textbox(
                        label="病历摘要",
                        lines=12,
                        placeholder=CLINICAL_NOTE_PLACEHOLDER,
                        elem_classes=["clinical-note-input"],
                    )

            readiness_html = gr.HTML(value=_init_readiness_html)
            confirm_overwrite = gr.Checkbox(
                label="覆盖已有草稿（再次生成须勾选）",
                value=False,
            )
            run_btn = gr.Button(
                "生成草稿",
                variant="primary",
                elem_classes=["primary"],
                interactive=_can_generate(None, ""),
            )
            gr.HTML(AI_DRAFT_LABEL_HTML)
            ai_draft = gr.Textbox(
                label=None,
                show_label=False,
                lines=18,
                elem_classes=["result-box"],
                placeholder="生成内容在此编辑",
            )
            raw_draft_state = gr.State("")
            status = gr.Textbox(
                label="状态",
                interactive=False,
                elem_classes=["status-strip"],
            )

            save_btn = gr.Button("保存病例", variant="secondary")
            save_status = gr.Textbox(label="保存", interactive=False)

        with gr.Tab("病例管理"):
            selected_case_id_state = gr.State("")
            case_ids_state = gr.State(_initial_case_ids)
            case_table = gr.Dataframe(
                value=_initial_df,
                interactive=False,
                label="列表",
                wrap=True,
            )
            with gr.Row():
                delete_confirm = gr.Checkbox(label="确认删除", value=False)
                delete_btn = gr.Button("删除", variant="stop", size="sm")
            loaded_note = gr.Textbox(
                label="病历摘要",
                lines=8,
                placeholder="点表加载",
            )
            loaded_ai = gr.Textbox(label="AI 草稿", lines=14, elem_classes=["result-box"])
            loaded_edit = gr.Textbox(label="医生稿", lines=14, elem_classes=["result-box"])
            load_status = gr.Textbox(label="加载", interactive=False)

    demo.load(fn=lambda: readiness_ui(None, ""), outputs=[readiness_html, run_btn])

    btn_note_template.click(
        fn=clinical_note_fill_template_ready,
        inputs=[image],
        outputs=[note, readiness_html, run_btn],
    )
    btn_note_clear.click(
        fn=clinical_note_clear_ready,
        inputs=[image],
        outputs=[note, readiness_html, run_btn],
    )

    image.change(fn=readiness_ui, inputs=[image, note], outputs=[readiness_html, run_btn])
    note.change(fn=readiness_ui, inputs=[image, note], outputs=[readiness_html, run_btn])

    run_btn.click(
        fn=infer,
        inputs=[image, note, ai_draft, confirm_overwrite],
        outputs=[ai_draft, raw_draft_state, status, confirm_overwrite],
        show_progress="full",
    )
    save_btn.click(
        fn=save_report,
        inputs=[image, note, raw_draft_state, ai_draft],
        outputs=[save_status, case_table, case_ids_state],
    )
    delete_btn.click(
        fn=delete_case_ui,
        inputs=[selected_case_id_state, delete_confirm],
        outputs=[
            case_table,
            case_ids_state,
            delete_confirm,
            selected_case_id_state,
            loaded_note,
            loaded_ai,
            loaded_edit,
            load_status,
        ],
    )
    case_table.select(
        fn=open_case,
        inputs=[case_ids_state],
        outputs=[loaded_note, loaded_ai, loaded_edit, load_status, selected_case_id_state],
    )

if __name__ == "__main__":
    port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    share = os.getenv("GRADIO_SHARE", "false").lower() in {"1", "true", "yes"}
    auth_user = os.getenv("APP_AUTH_USER", "")
    auth_pass = os.getenv("APP_AUTH_PASS", "")
    auth = (auth_user, auth_pass) if auth_user and auth_pass else None
    demo.launch(server_name="0.0.0.0", server_port=port, auth=auth, share=share, css=APP_CSS)
