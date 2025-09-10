import argparse
import json
import pathlib
import re
from typing import Dict, List, Tuple

from src.app.data_ingestor.vector_index import VectorIndex
from src.app.rag.orchestrator import RagOrchestrator


def _normalize_text(s: str) -> str:
    # Strip RAG "Sources:" footer if present
    s = re.split(r"\n\s*Sources:\s*", s, maxsplit=1)[0]
    # Lowercase and remove punctuation-like chars
    s = s.lower()
    s = re.sub(r"[\u2018\u2019\u201C\u201D]", "'", s)  # normalize quotes
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _tokenize(s: str) -> List[str]:
    return [t for t in _normalize_text(s).split(" ") if t]


def prf1(pred: str, gold: str) -> Tuple[float, float, float]:
    """Bag-of-words precision/recall/F1 computed with token counts."""
    ptoks = _tokenize(pred)
    gtoks = _tokenize(gold)
    if not ptoks and not gtoks:
        return 1.0, 1.0, 1.0
    if not ptoks:
        return 0.0, 0.0, 0.0
    if not gtoks:
        return 0.0, 0.0, 0.0

    from collections import Counter

    pc = Counter(ptoks)
    gc = Counter(gtoks)
    overlap = sum(min(pc[w], gc[w]) for w in pc.keys() | gc.keys())
    precision = overlap / max(1, sum(pc.values()))
    recall = overlap / max(1, sum(gc.values()))
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def evaluate(dataset_path: str, index_dir: str, oos_threshold: float = 0.2, provider: str = "none", api_key: str = "") -> Dict:
    vx = VectorIndex.load(index_dir)
    rag = RagOrchestrator(provider, api_key, vx, oos_threshold)

    ds = json.loads(pathlib.Path(dataset_path).read_text(encoding="utf-8"))

    results = []
    p_sum = r_sum = f_sum = 0.0
    for item in ds:
        q = item["question"]
        gold = item["expected_answer"]
        out = rag.get_grounded_response(q)
        pred = out.get("answer", "")
        p, r, f = prf1(pred, gold)
        p_sum += p; r_sum += r; f_sum += f
        results.append({
            "id": item.get("id"),
            "question": q,
            "expected_answer": gold,
            "predicted_answer": pred,
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f, 4),
            "citations": out.get("citations"),
        })

    n = max(1, len(results))
    summary = {
        "items": len(results),
        "avg_precision": round(p_sum / n, 4),
        "avg_recall": round(r_sum / n, 4),
        "avg_f1": round(f_sum / n, 4),
    }
    return {"summary": summary, "results": results}


def main():
    ap = argparse.ArgumentParser(description="Evaluate RAG answers vs expected text using token P/R/F1.")
    ap.add_argument("--dataset", default="kb/qa_test_dataset.json", help="Path to QA JSON dataset")
    ap.add_argument("--index", default="rag_index", help="Path to vector index directory")
    ap.add_argument("--oos_threshold", type=float, default=0.2, help="Out-of-scope score threshold [0-1]")
    ap.add_argument("--provider", default="none", help="LLM provider (e.g., 'gemini' or 'none')")
    ap.add_argument("--api_key", default="", help="API key if provider requires it")
    ap.add_argument("--out", default="rag_index/qa_eval_report.json", help="Optional path to write JSON report")
    args = ap.parse_args()

    report = evaluate(args.dataset, args.index, args.oos_threshold, args.provider, args.api_key)

    # Pretty print summary
    s = report["summary"]
    print(f"Items: {s['items']}")
    print(f"Avg Precision: {s['avg_precision']:.4f}")
    print(f"Avg Recall:    {s['avg_recall']:.4f}")
    print(f"Avg F1:        {s['avg_f1']:.4f}")

    # Save detailed report
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote report to: {out_path}")


if __name__ == "__main__":
    main()

