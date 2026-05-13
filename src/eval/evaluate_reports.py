from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch.nn.functional as F
from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU

from core.report_schema import parse_structured_report

_DEFAULT_SEMANTIC_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def _rouge_avg(refs: list[str], hyps: list[str]) -> dict[str, float]:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    r1: list[float] = []
    r2: list[float] = []
    rl: list[float] = []
    for ref, hyp in zip(refs, hyps):
        s = scorer.score(ref, hyp)
        r1.append(s["rouge1"].fmeasure)
        r2.append(s["rouge2"].fmeasure)
        rl.append(s["rougeL"].fmeasure)
    n = len(r1)
    if n == 0:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    return {"rouge1": sum(r1) / n, "rouge2": sum(r2) / n, "rougeL": sum(rl) / n}


def _bleu_corpus(refs: list[str], hyps: list[str]) -> dict[str, float | int]:
    bleu = BLEU(effective_order=True)
    sc = bleu.corpus_score(hyps, [refs])
    # sacrebleu 為 0–100，與常見 HuggingFace evaluate JSON（0–1）對齊
    return {
        "bleu": sc.score / 100.0,
        "translation_length": sc.sys_len,
        "reference_length": sc.ref_len,
    }


def _semantic_cosine_avg(refs: list[str], hyps: list[str], model_id: str) -> dict[str, float | str]:
    """多語句向量餘弦相似度（逐樣本對齊後平均），適用英/中文本對齊評估。"""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_id)
    r2 = [r if r.strip() else " " for r in refs]
    h2 = [h if h.strip() else " " for h in hyps]
    er = model.encode(
        r2,
        convert_to_tensor=True,
        show_progress_bar=False,
        batch_size=min(32, max(1, len(r2))),
    )
    eh = model.encode(
        h2,
        convert_to_tensor=True,
        show_progress_bar=False,
        batch_size=min(32, max(1, len(h2))),
    )
    er = F.normalize(er, p=2, dim=1)
    eh = F.normalize(eh, p=2, dim=1)
    sims = (er * eh).sum(dim=1)
    return {
        "model": model_id,
        "cosine_similarity_avg": float(sims.mean().cpu()),
        "cosine_similarity_min": float(sims.min().cpu()),
        "cosine_similarity_max": float(sims.max().cpu()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate report generation quality")
    parser.add_argument("--pred", default="output/demo_predictions.jsonl")
    parser.add_argument("--out", default="output/report_metrics.json")
    parser.add_argument(
        "--no-semantic",
        action="store_true",
        help="跳過多語語義相似度（無需下載句向量模型）",
    )
    parser.add_argument(
        "--semantic-model",
        default=os.environ.get("REPORT_SEMANTIC_MODEL", _DEFAULT_SEMANTIC_MODEL),
        help="sentence-transformers 模型 id（預設可由環境變數 REPORT_SEMANTIC_MODEL 覆寫）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = [json.loads(x) for x in Path(args.pred).read_text(encoding="utf-8").splitlines() if x.strip()]
    refs = [r.get("target_report", "") for r in rows]
    hyps = [r.get("prediction", "") for r in rows]

    rouge_res = _rouge_avg(refs, hyps)
    bleu_res = _bleu_corpus(refs, hyps)

    completeness = []
    for h in hyps:
        completeness.append(parse_structured_report(h).completeness)

    result: dict = {
        "n_samples": len(rows),
        "rouge": rouge_res,
        "bleu": bleu_res,
        "structured_completeness_avg": sum(completeness) / len(completeness) if completeness else 0.0,
    }

    if not args.no_semantic:
        result["semantic_multilingual"] = _semantic_cosine_avg(refs, hyps, args.semantic_model)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
