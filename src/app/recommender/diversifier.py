
from typing import List
import numpy as np

class Diversifier:
    lam: float  # REC_LAMBDA

    def __init__(self, lam: float = 0.7):
        self.lam = lam

    def select(self, scores: np.ndarray, embs: np.ndarray, k: int=3) -> List[int]:
        n = embs.shape[0]
        if n == 0:
            return []
        picked: List[int] = []
        cand = set(range(n))
        while cand and len(picked) < k:
            best_i, best_val = None, -1e9
            for i in list(cand):
                div = max(float(embs[i] @ embs[j]) for j in picked) if picked else 0.0
                val = self.lam * float(scores[i]) - (1.0 - self.lam) * div
                if val > best_val:
                    best_val, best_i = val, i
            picked.append(best_i)
            cand.remove(best_i)  
        return picked