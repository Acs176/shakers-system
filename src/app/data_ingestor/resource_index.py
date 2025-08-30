from typing import List, Dict
import numpy as np
import json
from src.app.recommender.vectorizer import Vectorizer

class ResourceIndex:
    def __init__(self, items: List[Dict], model_name: str):
        self.items = items
        self.model_name = model_name
        self._emb = None
        self._id2idx = {r["id"]: i for i, r in enumerate(items)}

    def ensure_embeddings(self, vectorizer: Vectorizer):
        if self._emb is None:
            texts = [
                f"{r.get('title','')} — {', '.join(r.get('tags', []))}. {r.get('description','')}"
                for r in self.items
            ]
            self._emb = vectorizer.embed_texts(texts)
            # defensive normalize
            norms = np.linalg.norm(self._emb, axis=1, keepdims=True) + 1e-12
            self._emb = self._emb / norms

    @property
    def emb(self) -> np.ndarray:
        if self._emb is None:
            raise RuntimeError("Call ensure_embeddings(vectorizer) first.")
        return self._emb
    
def load_resource_index(
    resource_json_path: str, 
    model_name: str="sentence-transformers/all-MiniLM-L6-v2"
) -> ResourceIndex:
    with open(resource_json_path, "r", encoding="utf-8") as f:
        items = json.load(f)
    return ResourceIndex(items, model_name=model_name)