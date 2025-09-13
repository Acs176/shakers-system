from __future__ import annotations

from typing import Iterable, Sequence, Set


def _first_relevant_rank(relevant: Set[str], ranked: Sequence[str]) -> int | None:
    for i, x in enumerate(ranked, start=1):
        if x in relevant:
            return i
    return None


def precision_at_k(relevant: Set[str], ranked: Sequence[str], k: int) -> float:
    if k <= 0:
        return 0.0
    topk = list(ranked[:k])
    hits = sum(1 for x in topk if x in relevant)
    return float(hits) / float(k)


def recall_at_k(relevant: Set[str], ranked: Sequence[str], k: int) -> float:
    if not relevant:
        # If nothing is labeled relevant, define perfect recall
        return 1.0
    topk = set(ranked[:k])
    hits = len(relevant & topk)
    return float(hits) / float(len(relevant))


def hit_at_k(relevant: Set[str], ranked: Sequence[str], k: int) -> float:
    topk = set(ranked[:k])
    return 1.0 if (relevant & topk) else 0.0


def mrr_at_k(relevant: Set[str], ranked: Sequence[str], k: int) -> float:
    r = _first_relevant_rank(relevant, ranked[:k])
    return 0.0 if r is None else 1.0 / float(r)


def dcg_at_k(relevant: Set[str], ranked: Sequence[str], k: int) -> float:
    # Binary gains: rel_i in {0,1}
    import math

    dcg = 0.0
    for i, x in enumerate(ranked[:k], start=1):
        gain = 1.0 if x in relevant else 0.0
        if i == 1:
            dcg += gain
        else:
            dcg += gain / math.log2(i)
    return dcg


def idcg_at_k(num_relevant: int, k: int) -> float:
    # Ideal DCG with binary gains: all relevant ranked first
    import math

    R = min(num_relevant, k)
    if R <= 0:
        return 0.0
    idcg = 1.0  # i=1
    for i in range(2, R + 1):
        idcg += 1.0 / math.log2(i)
    return idcg


def ndcg_at_k(relevant: Set[str], ranked: Sequence[str], k: int) -> float:
    idcg = idcg_at_k(len(relevant), k)
    if idcg == 0.0:
        return 1.0  # no relevant labels -> define as perfect
    dcg = dcg_at_k(relevant, ranked, k)
    return float(dcg) / float(idcg)

