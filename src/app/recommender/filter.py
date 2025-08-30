from typing import List, Dict, Sequence
import numpy as np


class Filter:
    def filter_seen(self, items: Sequence[Dict], seen_ids: Sequence[str]) -> List[int]:
        seen = set(seen_ids or [])
        return [i for i, r in enumerate(items) if r["id"] not in seen]

    def apply_threshold(self, scores: np.ndarray, thresh: float) -> np.ndarray:
        return scores >= thresh



