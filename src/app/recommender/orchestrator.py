from typing import List, Dict

from src.app.recommender.diversifier import Diversifier
from src.app.recommender.explainer import Explainer
from src.app.recommender.filter import Filter
from src.app.recommender.scorer import Scorer
from src.app.recommender.vectorizer import Vectorizer
from src.app.data_ingestor.resource_index import ResourceIndex
from src.logging_setup import span

class Recommender:
    vectorizer: Vectorizer
    scorer: Scorer
    filter: Filter
    diversifier: Diversifier
    explainer: Explainer
    index: ResourceIndex
    rec_thresh: float

    def __init__(self, ri: ResourceIndex, rec_thresh: float = 0.35):
        self.index = ri
        self.vectorizer = Vectorizer()
        self.scorer = Scorer()
        self.filter = Filter()
        self.diversifier = Diversifier()
        self.explainer = Explainer()
        self.rec_thresh = rec_thresh

    def recommend(self, profile, query: str) -> List[Dict]:
        with span("recommend.run"):
            self.index.ensure_embeddings(self.vectorizer)

            cand_idxs = self.filter.filter_seen(self.index.items, getattr(profile, "seen_resource_ids", []))
            if not cand_idxs:
                return []

            E = self.index.emb[cand_idxs]                # (m, d)
            v = self.vectorizer.intent_vector(profile, query)  # (d,)
            rel = self.scorer.score(v, E)           # (m,)

            mask = self.filter.apply_threshold(rel, self.rec_thresh)
            if not mask.any():
                return []

            # remove below threshold
            pass_idxs = [cand_idxs[i] for i, keep in enumerate(mask) if keep]
            E_pass = E[mask]
            rel_pass = rel[mask]

            # TODO: Maybe apply diversification before threshold
            picked_local = self.diversifier.select(rel_pass, E_pass)

            # format output
            out = []
            for pi in picked_local:
                gi = pass_idxs[pi]
                r = self.index.items[gi]
                out.append({
                    "id": r["id"],
                    "title": r["title"],
                    "url": r.get("url"),
                    "tags": r.get("tags", []),
                    "why": self.explainer.why(r, profile, query),
                    "score": round(float(rel_pass[pi]), 3),
                })
            return out


    