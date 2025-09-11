import re
import sacrebleu
from rouge_score import rouge_scorer as _rouge_scorer
from typing import List, Tuple, Dict

def strip_sources_footer(s: str) -> str:
    return re.split(r"\n\s*Sources:\s*", s, maxsplit=1)[0]


def _normalize_text(s: str) -> str:
    # Strip RAG "Sources:" footer if present
    s = strip_sources_footer(s)
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

_ROUGE_SCORER = None
def sentence_bleu(pred: str, gold: str) -> float:
    """Sentence BLEU via sacrebleu; returns 0..1. Requires sacrebleu."""
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    # sacrebleu returns percentage 0..100
    return float(sacrebleu.sentence_bleu(pred, [gold]).score) / 100.0


def rouge_scores(pred: str, gold: str) -> Dict[str, float]:
    """ROUGE-1/2/L F1 via rouge-score; returns values 0..1. Requires rouge-score."""
    if not pred and not gold:
        return {"rouge1": 1.0, "rouge2": 1.0, "rougeL": 1.0}
    if not pred or not gold:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    global _ROUGE_SCORER
    if _ROUGE_SCORER is None:
        _ROUGE_SCORER = _rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    # RougeScorer.score expects target (reference), prediction
    sc = _ROUGE_SCORER.score(gold, pred)
    return {
        "rouge1": float(sc["rouge1"].fmeasure),
        "rouge2": float(sc["rouge2"].fmeasure),
        "rougeL": float(sc["rougeL"].fmeasure),
    }