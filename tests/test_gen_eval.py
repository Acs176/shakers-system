import json
import pathlib

from src.eval.generation_eval import evaluate


def test_rag_precision_recall_smoke():
    ds_path = "kb/gen_test_dataset.json"
    idx_dir = "rag_index"

    # Run evaluation using extractive fallback (no external LLM calls)
    report = evaluate(ds_path, idx_dir, oos_threshold=0.2, provider="none", api_key="")

    assert "summary" in report and "results" in report
    summary = report["summary"]
    results = report["results"]

    # Basic shape checks
    assert summary["items"] == len(results) > 0
    for r in results:
        assert "precision" in r and "recall" in r and "f1" in r
        assert 0.0 <= r["precision"] <= 1.0
        assert 0.0 <= r["recall"] <= 1.0
        assert 0.0 <= r["f1"] <= 1.0

