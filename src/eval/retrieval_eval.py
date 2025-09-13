import argparse
import json
import pathlib
from typing import Dict, List, Set, Tuple

from src.app.data_ingestor.vector_index import VectorIndex
from src.app.rag.retriever import Retriever
from src.eval.retrieval_metrics import (
    precision_at_k,
    recall_at_k,
    hit_at_k,
    mrr_at_k,
    ndcg_at_k,
)


def _norm_source_name(p: str) -> str:
    return pathlib.Path(p).name


def _relevant_ids(item: Dict, granularity: str) -> Set[str]:
    if granularity == "chunk":
        # Expect explicit chunk IDs in dataset if using chunk granularity
        return set(item.get("relevant_chunks", []))
    # default: document-level relevance from `source_docs`
    return set(_norm_source_name(s) for s in item.get("source_docs", []))


def _ranked_ids(retrieved: List[Dict], granularity: str) -> List[str]:
    if granularity == "chunk":
        return [d.get("id", "") for d in retrieved]
    return [d.get("source", "") for d in retrieved]


def evaluate(
    dataset_path: str,
    index_dir: str,
    k: int = 5,
    oos_threshold: float = 0.2,
    granularity: str = "doc",
) -> Dict:
    if granularity not in {"doc", "chunk"}:
        raise ValueError("granularity must be 'doc' or 'chunk'")

    vx = VectorIndex.load(index_dir)
    retriever = Retriever(vx, oos_threshold)

    ds = json.loads(pathlib.Path(dataset_path).read_text(encoding="utf-8"))

    results = []
    prec_sum = rec_sum = hit_sum = mrr_sum = ndcg_sum = 0.0
    oos_total = 0

    for item in ds:
        q = item["question"]
        rel = _relevant_ids(item, granularity)

        top_docs, out_of_scope, citations, _ = retriever.get_top_results(q, k=k, return_texts=False)
        if out_of_scope:
            oos_total += 1

        ranked = _ranked_ids(top_docs, granularity)

        prec = precision_at_k(rel, ranked, k)
        rec = recall_at_k(rel, ranked, k)
        hit = hit_at_k(rel, ranked, k)
        mrr = mrr_at_k(rel, ranked, k)
        ndcg = ndcg_at_k(rel, ranked, k)
        prec_sum += prec; rec_sum += rec; hit_sum += hit; mrr_sum += mrr; ndcg_sum += ndcg

        detailed = []
        for rank, d in enumerate(top_docs, start=1):
            det_key = d.get("id") if granularity == "chunk" else d.get("source")
            detailed.append({
                "rank": rank,
                "source": d.get("source"),
                "id": d.get("id"),
                "score": round((float(d.get("score", 0.0)) + 1.0) / 2.0, 3),  # [-1,1] -> [0,1]
                "is_relevant": bool(det_key in rel),
                "title": d.get("title"),
                "section": d.get("section"),
            })

        results.append({
            "id": item.get("id"),
            "question": q,
            "source_docs": item.get("source_docs", []),
            "relevant_ids": sorted(list(rel)),
            "granularity": granularity,
            "out_of_scope": bool(out_of_scope),
            "metrics": {
                "precision@k": round(prec, 4),
                "recall@k": round(rec, 4),
                "hit@k": round(hit, 4),
                "mrr@k": round(mrr, 4),
                "ndcg@k": round(ndcg, 4),
            },
            "retrieved": detailed,
        })

    n = max(1, len(results))
    summary = {
        "items": len(results),
        "k": k,
        "granularity": granularity,
        "avg_precision@k": round(prec_sum / n, 4),
        "avg_recall@k": round(rec_sum / n, 4),
        "avg_hit@k": round(hit_sum / n, 4),
        "avg_mrr@k": round(mrr_sum / n, 4),
        "avg_ndcg@k": round(ndcg_sum / n, 4),
        "oos_rate": round(float(oos_total) / float(n), 4),
        "oos_count": oos_total,
    }

    return {"summary": summary, "results": results}


def main():
    ap = argparse.ArgumentParser(description="Evaluate RAG retriever quality (Recall@k, MRR, nDCG, etc.)")
    ap.add_argument("--dataset", default="kb/retrieval_eval_dataset.json", help="Path to JSON dataset")
    ap.add_argument("--index", default="rag_index", help="Path to vector index directory")
    ap.add_argument("--k", type=int, default=5, help="Top-k to retrieve")
    ap.add_argument("--oos_threshold", type=float, default=0.2, help="Out-of-scope score threshold [0-1]")
    ap.add_argument("--granularity", choices=["doc", "chunk"], default="doc", help="Relevance granularity")
    ap.add_argument("--out", default="rag_index/retrieval_eval_report.json", help="JSON report path")
    # Optional CI thresholds
    ap.add_argument("--recall_min", type=float, default=None, help="Fail unless avg_recall@k > value")
    ap.add_argument("--mrr_min", type=float, default=None, help="Fail unless avg_mrr@k > value")
    ap.add_argument("--ndcg_min", type=float, default=None, help="Fail unless avg_ndcg@k > value")
    args = ap.parse_args()

    report = evaluate(
        dataset_path=args.dataset,
        index_dir=args.index,
        k=args.k,
        oos_threshold=args.oos_threshold,
        granularity=args.granularity,
    )

    s = report["summary"]
    print(f"Items:        {s['items']}")
    print(f"k:            {s['k']}")
    print(f"Granularity:  {s['granularity']}")
    print(f"Avg P@k:      {s['avg_precision@k']:.4f}")
    print(f"Avg R@k:      {s['avg_recall@k']:.4f}")
    print(f"Avg Hit@k:    {s['avg_hit@k']:.4f}")
    print(f"Avg MRR@k:    {s['avg_mrr@k']:.4f}")
    print(f"Avg nDCG@k:   {s['avg_ndcg@k']:.4f}")
    print(f"OOS rate:     {s['oos_rate']:.4f} ({s['oos_count']} of {s['items']})")

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote report to: {out_path}")

    # Threshold enforcement
    def _check(name: str, value: float | None, actual: float) -> bool:
        if value is None:
            return True
        ok = float(actual) > float(value)
        print(f"Check {name} > {value:.4f}: {actual:.4f} -> {'OK' if ok else 'FAIL'}")
        return ok

    ok = True
    ok &= _check("avg_recall@k", args.recall_min, s["avg_recall@k"]) if args.recall_min is not None else True
    ok &= _check("avg_mrr@k", args.mrr_min, s["avg_mrr@k"]) if args.mrr_min is not None else True
    ok &= _check("avg_ndcg@k", args.ndcg_min, s["avg_ndcg@k"]) if args.ndcg_min is not None else True
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

