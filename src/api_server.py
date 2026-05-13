from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from core.case_store import append_case, get_case, list_cases
from core.report_schema import SECTION_ORDER
from inference import QwenVLMReporter

app = FastAPI(title="Lightweight Med API", version="0.2.0")
reporter: QwenVLMReporter | None = None


class InferRequest(BaseModel):
    image_path: str = Field(..., description="Absolute image path")
    clinical_note: str = Field(..., min_length=1)


class SaveRequest(BaseModel):
    image_path: str
    clinical_note: str
    ai_draft: str
    doctor_edited_report: str


def get_reporter() -> QwenVLMReporter:
    global reporter
    if reporter is None:
        reporter = QwenVLMReporter()
    return reporter


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/infer")
def infer(req: InferRequest) -> dict:
    image_file = Path(req.image_path)
    if not image_file.exists():
        raise HTTPException(status_code=404, detail=f"image not found: {image_file}")

    model = get_reporter()
    structured = model.generate_structured(req.image_path, req.clinical_note)
    return {
        "raw_text": structured.raw_text,
        "sections": structured.sections,
        "section_order": SECTION_ORDER,
        "completeness": structured.completeness,
    }


@app.post("/cases")
def save_case(req: SaveRequest) -> dict:
    payload = append_case(req.model_dump())
    return payload


@app.get("/cases")
def get_cases(limit: int = 100) -> dict:
    rows = list_cases(limit=limit)
    return {"total": len(rows), "items": rows}


@app.get("/cases/{case_id}")
def get_case_detail(case_id: str) -> dict:
    row = get_case(case_id)
    if not row:
        raise HTTPException(status_code=404, detail="case not found")
    return row
