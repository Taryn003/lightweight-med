from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

CASES_FILE = Path("output/doctor_edited_reports.jsonl")


def ensure_output() -> None:
    CASES_FILE.parent.mkdir(parents=True, exist_ok=True)


def _read_all_records() -> list[dict[str, Any]]:
    if not CASES_FILE.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in CASES_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _sort_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda r: r.get("saved_at", ""), reverse=True)


def _write_all_records(rows: list[dict[str, Any]]) -> None:
    ensure_output()
    with CASES_FILE.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def append_case(record: dict[str, Any]) -> dict[str, Any]:
    ensure_output()
    payload = {
        "case_id": record.get("case_id") or str(uuid.uuid4()),
        "saved_at": record.get("saved_at") or datetime.now().isoformat(timespec="seconds"),
        **record,
    }
    with CASES_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return payload


def list_cases(limit: int = 200) -> list[dict[str, Any]]:
    rows = _sort_records(_read_all_records())
    return rows[:limit]


def list_cases_filtered(limit: int = 10_000) -> list[dict[str, Any]]:
    """全量记录（按保存时间倒序），用于分页与导出。"""
    rows = _sort_records(_read_all_records())
    return rows[:limit]


def get_case(case_id: str) -> dict[str, Any] | None:
    for row in list_cases_filtered(limit=50_000):
        if row.get("case_id") == case_id:
            return row
    return None


def delete_case(case_id: str, *, remove_image: bool = True) -> bool:
    """按 case_id 删除一条记录；成功返回 True。"""
    cid = (case_id or "").strip()
    if not cid:
        return False
    case = get_case(cid)
    rows = _read_all_records()
    new_rows = [r for r in rows if r.get("case_id") != cid]
    if len(new_rows) == len(rows):
        return False
    _write_all_records(_sort_records(new_rows))
    if remove_image and case:
        p = case.get("image_path")
        if p:
            try:
                ip = Path(str(p))
                if ip.is_file() and "case_images" in ip.parts:
                    ip.unlink()
            except OSError:
                pass
    return True


def export_cases_snapshot() -> Path:
    """复制当前病例库为带时间戳的 JSONL，返回目标路径。"""
    ensure_output()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = CASES_FILE.parent / f"cases_export_{ts}.jsonl"
    if CASES_FILE.exists():
        shutil.copy2(CASES_FILE, dest)
    else:
        dest.write_text("", encoding="utf-8")
    return dest.resolve()
