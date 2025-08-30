import numpy as np
from typing import List, Optional

class Vectorizer:
    model_name: str
    alpha: float

    def __init__(self, model_name: str="sentence-transformers/all-MiniLM-L6-v2", alpha: float = 0.7):
        self.model_name = model_name
        self.alpha = alpha

    def _load_model(self):
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(self.model_name)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)
        model = self._load_model()
        vecs = model.encode(texts, normalize_embeddings=True)
        return np.asarray(vecs, dtype=np.float32)

    def history_centroid(self, profile, last_n: int = 20) -> Optional[np.ndarray]:
        if profile is None or len(profile.query_history) == 0:
            return None
        qs = [e.text for e in profile.query_history][-last_n:]
        if not qs:
            return None
        V = self.embed_texts(qs)
        if V.size == 0:
            return None
        c = V.mean(axis=0)
        return c / (np.linalg.norm(c) + 1e-12)

    def intent_vector(self, profile, user_query: str) -> np.ndarray:
        e_q = self.embed_texts([user_query]).reshape(-1)
        c = self.history_centroid(profile)
        if c is None:
            return e_q
        v = self.alpha * e_q + (1.0 - self.alpha) * c
        return v / (np.linalg.norm(v) + 1e-12)