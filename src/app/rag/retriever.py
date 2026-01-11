from typing import List, Dict, Optional, Tuple
from src.app.data_ingestor.vector_index import VectorStore

class Retriever:
    vector_store: VectorStore
    
    def __init__(self, vx, oos_threshold, METRICS=None):
        self.METRICS = METRICS
        self.oos_threshold = oos_threshold
        self.vector_store = vx

    def get_top_results(self, query: str, k: int = 3, return_texts: bool = False) -> Tuple[List[Dict], bool, Optional[List[Dict]], Optional[List[str]]]:

        top_docs = self.top_k_search(query, k)
        out_of_scope = False
        if not top_docs:
            out_of_scope = True
            if self.METRICS:
                self.METRICS["out_of_scope_total"].add(1)
        else:
            max_sim = max(d["score"] for d in top_docs)
            # Convert inner product (cosine) [-1,1] to [0,1] if you want a human-friendly score
            max_sim01 = (max_sim + 1.0) / 2.0
            out_of_scope = max_sim01 < self.oos_threshold

        if out_of_scope:
            return ([], out_of_scope, None, [] if return_texts else None)
        else:
            citations = [{
                "title": d["title"],
                "section": d["section"],
                "source": d["source"],
                "chunk_id": d["id"],
                "score": round((d["score"] + 1.0) / 2.0, 3)
            } for d in top_docs]
            ctx_texts = [d["text"] for d in top_docs] if return_texts else None
            return top_docs, out_of_scope, citations, ctx_texts


    def top_k_search(self, query: str, k: int = 3) -> List[Dict]:
        """
        returns a dictionary with the content and score of each top-k
        retrieved document
        """
        return self.vector_store.search(query, k)
