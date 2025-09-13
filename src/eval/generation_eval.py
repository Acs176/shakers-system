import argparse
import json
import pathlib
from typing import Dict

from src.app.data_ingestor.vector_index import VectorIndex
from src.app.rag.orchestrator import RagOrchestrator
from src.eval.ragas_faithfulness import compute_faithfulness
from src.eval.syntactic import prf1, strip_sources_footer, rouge_scores, sentence_bleu
from src.eval.semantic import semantic_similarity


def evaluate(
    dataset_path: str, 
    index_dir: str, 
    oos_threshold: float = 0.2, 
    provider: str = "none", 
    api_key: str = "", 
    sim_model: str = "sentence-transformers/all-mpnet-base-v2", 
    with_faithfulness: bool = False, 
    ) -> Dict:
    
    vx = VectorIndex.load(index_dir)
    rag = RagOrchestrator(provider, api_key, vx, oos_threshold)

    ds = json.loads(pathlib.Path(dataset_path).read_text(encoding="utf-8"))

    results = []
    p_sum = r_sum = f_sum = s_sum = b_sum = r1_sum = r2_sum = rl_sum = 0.0
    faith_sum = 0.0
    faith_count = 0
    for item in ds:
        q = item["question"]
        gold = item["expected_answer"]
        out = rag.get_grounded_response(q, include_contexts=with_faithfulness)
        pred = out.get("answer", "")
        # Use answer body (strip 'Sources:' footer) for text metrics
        pred_body = strip_sources_footer(pred)
        p, r, f = prf1(pred_body, gold)
        s = semantic_similarity(pred_body, gold, model_name=sim_model)
        b = sentence_bleu(pred_body, gold)
        rouge = rouge_scores(pred_body, gold)
        faith = None
        if with_faithfulness:
            try:
                contexts = out.get("contexts") or []
                if contexts:
                    faith = compute_faithfulness(q, pred_body, contexts, provider=provider, api_key=(api_key or None))
                    faith_sum += float(faith)
                    faith_count += 1
            except Exception as e:
                faith = None
        p_sum += p; r_sum += r; f_sum += f; s_sum += s; b_sum += b
        r1_sum += rouge["rouge1"]; r2_sum += rouge["rouge2"]; rl_sum += rouge["rougeL"]
        results.append({
            "id": item.get("id"),
            "question": q,
            "expected_answer": gold,
            "predicted_answer": pred,
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f, 4),
            "semantic": round(s, 4),
            "bleu": round(b, 4),
            "rouge1": round(rouge["rouge1"], 4),
            "rouge2": round(rouge["rouge2"], 4),
            "rougeL": round(rouge["rougeL"], 4),
            "faithfulness": None if faith is None else round(float(faith), 4),
            "citations": out.get("citations"),
            "contexts": out.get("contexts") if with_faithfulness else None,
        })

    n = max(1, len(results))
    summary = {
        "items": len(results),
        "avg_precision": round(p_sum / n, 4),
        "avg_recall": round(r_sum / n, 4),
        "avg_f1": round(f_sum / n, 4),
        "avg_semantic": round(s_sum / n, 4),
        "avg_bleu": round(b_sum / n, 4),
        "avg_rouge1": round(r1_sum / n, 4),
        "avg_rouge2": round(r2_sum / n, 4),
        "avg_rougeL": round(rl_sum / n, 4),
        "avg_faithfulness": None if faith_count == 0 else round(faith_sum / faith_count, 4),
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
    ap.add_argument("--sim_model", default="sentence-transformers/all-mpnet-base-v2", help="SentenceTransformer model name for semantic similarity")
    ap.add_argument("--with_faithfulness", action="store_true", help="Compute RAGAS faithfulness (requires API key and internet)")
    ap.add_argument("--semantic_min", type=float, default=None, help="If set, exit non-zero unless avg_semantic > this value")
    ap.add_argument("--faithfulness_min", type=float, default=None, help="If set with --with_faithfulness, exit non-zero unless avg_faithfulness > this value")
    args = ap.parse_args()

    report = evaluate(
        args.dataset,
        args.index,
        args.oos_threshold,
        args.provider,
        args.api_key,
        args.sim_model,
        args.with_faithfulness,
    )

    # Pretty print summary
    s = report["summary"]
    print(f"Items: {s['items']}")
    print(f"Avg Precision: {s['avg_precision']:.4f}")
    print(f"Avg Recall:    {s['avg_recall']:.4f}")
    print(f"Avg F1:        {s['avg_f1']:.4f}")
    print(f"Avg Semantic:  {s['avg_semantic']:.4f}")
    print(f"Avg BLEU:      {s['avg_bleu']:.4f}")
    print(f"Avg ROUGE-1:   {s['avg_rouge1']:.4f}")
    print(f"Avg ROUGE-2:   {s['avg_rouge2']:.4f}")
    print(f"Avg ROUGE-L:   {s['avg_rougeL']:.4f}")
    if s.get("avg_faithfulness") is not None:
        print(f"Avg Faithful.: {s['avg_faithfulness']:.4f}")

    # Save detailed report
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote report to: {out_path}")

    # Optional threshold enforcement for CI
    sem_min = args.semantic_min
    faith_min = args.faithfulness_min if args.with_faithfulness else None
    if sem_min is not None or faith_min is not None:
        ok = True
        avg_sem = s.get("avg_semantic", 0.0)
        if sem_min is not None:
            sem_ok = float(avg_sem) > float(sem_min)
            print(f"Check avg_semantic > {sem_min:.4f}: {avg_sem:.4f} -> {'OK' if sem_ok else 'FAIL'}")
            ok = ok and sem_ok
        if faith_min is not None:
            avg_faith = s.get("avg_faithfulness")
            faith_ok = (avg_faith is not None) and (float(avg_faith) > float(faith_min))
            print(f"Check avg_faithfulness > {faith_min:.4f}: {avg_faith} -> {'OK' if faith_ok else 'FAIL'}")
            ok = ok and faith_ok
        if not ok:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
