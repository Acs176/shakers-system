import numpy as np

class Scorer:
    def score(self, query_vec: np.ndarray, item_embs: np.ndarray) -> np.ndarray:
        if item_embs.size == 0:
            return np.zeros((0,), dtype=np.float32)
        # assume normalized; then cosine = dot
        return item_embs @ query_vec