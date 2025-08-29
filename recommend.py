# -----------------------------
# --------- RECOMMENDATIONS ---
# -----------------------------
from typing import List, Dict, Optional
import json, time
import numpy as np
import uuid

# ---- Minimal knobs
REC_ALPHA = 0.7        # weight for current query vs. history centroid
REC_LAMBDA = 0.7       # relevance vs. diversity tradeoff for MMR
REC_THRESH = 0.35      # min cosine(sim) to consider a candidate
REC_TOPK = 3           # how many recs to return (2-3 recommended)

# ---- Embeddings
def _load_sbert(model_name: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)

def _embed_texts(model_name: str, texts: List[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)  # default width; will be replaced
    model = _load_sbert(model_name)
    vecs = model.encode(texts, normalize_embeddings=True)
    return np.asarray(vecs, dtype=np.float32)

def _cosine_matrix(vec: np.ndarray, mat: np.ndarray) -> np.ndarray:
    # vec: (d,), mat: (n,d) -> (n,)
    if mat.size == 0:
        return np.zeros((0,), dtype=np.float32)
    return mat @ vec  # already normalized

# ---- Resource catalog handling
class ResourceIndex:
    def __init__(self, items: List[Dict], model_name: str):
        self.items = items
        self.model_name = model_name
        self._emb = None  # lazy
        self._id2idx = {r["id"]: i for i, r in enumerate(items)}

    def ensure_embeddings(self):
        if self._emb is None:
            texts = [
                f"{r.get('title','')} — {', '.join(r.get('tags', []))}. {r.get('description','')}"
                for r in self.items
            ]
            self._emb = _embed_texts(self.model_name, texts)
            # normalize defensively (if model didn't)
            norms = np.linalg.norm(self._emb, axis=1, keepdims=True) + 1e-12
            self._emb = self._emb / norms

    @property
    def emb(self) -> np.ndarray:
        self.ensure_embeddings()
        return self._emb

def load_resource_index(resource_json_path: str, model_name: str) -> ResourceIndex:
    with open(resource_json_path, "r", encoding="utf-8") as f:
        items = json.load(f)
    return ResourceIndex(items, model_name=model_name)

# ---- User profile utilities
def _history_centroid(profile: Dict, model_name: str, last_n: int = 20) -> Optional[np.ndarray]:
    if profile is None:
        return None
    qs = [e["text"] for e in (profile.get("query_history") or [])][-last_n:]
    if not qs:
        return None
    V = _embed_texts(model_name, qs)
    if V.size == 0:
        return None
    c = V.mean(axis=0)
    c = c / (np.linalg.norm(c) + 1e-12)
    return c

def _primary_tag(resource: Dict) -> Optional[str]:
    tags = resource.get("tags") or []
    return tags[0] if tags else None

def _mmr_select(scores: np.ndarray, emb_mat: np.ndarray, k: int) -> List[int]:
    """
    Simple MMR over resource embeddings.
    scores: relevance scores (n,)
    emb_mat: normalized embeddings (n,d)
    """
    n = emb_mat.shape[0]
    if n == 0:
        return []
    picked: List[int] = []
    cand = set(range(n))
    while cand and len(picked) < k:
        best_i, best_score = None, -1e9
        for i in list(cand):
            div = 0.0
            if picked:
                # max cosine to already picked
                div = max(float(emb_mat[i] @ emb_mat[j]) for j in picked)
            s = REC_LAMBDA * float(scores[i]) - (1.0 - REC_LAMBDA) * div
            if s > best_score:
                best_score, best_i = s, i
        picked.append(best_i)
        cand.remove(best_i)
    return picked

def _why_for(resource: Dict, profile: Dict, user_query: str) -> str:
    tags = resource.get("tags") or []
    top_tags = set(profile.get("top_tags") or [])
    if top_tags and any(t in top_tags for t in tags):
        t = next(t for t in tags if t in top_tags)
        return f"Because you've been exploring **{t}**, this digs deeper via “{resource.get('title','')}”."
    # fallback to query match
    q_kw = user_query.strip()[:60]
    return f"Directly related to your question “{q_kw}…”, and complements your recent activity."

def recommend_resources(
    user_profile: Dict,
    user_query: str,
    res_index: ResourceIndex,
    k: int = REC_TOPK
) -> List[Dict]:
    """
    Returns up to k recommendations:
      [{id, title, url, tags, why, score}]
    """
    model_name = res_index.model_name
    e_q = _embed_texts(model_name, [user_query]).reshape(-1)
    c = _history_centroid(user_profile, model_name)
    if c is None:
        v = e_q
    else:
        v = REC_ALPHA * e_q + (1.0 - REC_ALPHA) * c
        v = v / (np.linalg.norm(v) + 1e-12)

    seen = set(user_profile.get("seen_resource_ids") or [])
    candidates = [(i, r) for i, r in enumerate(res_index.items) if r["id"] not in seen]

    if not candidates:
        return []

    idxs = np.array([i for i, _ in candidates], dtype=np.int32)
    E = res_index.emb[idxs]
    rel = _cosine_matrix(v, E)  # (m,)

    # threshold
    mask = rel >= REC_THRESH
    if not mask.any():
        return []

    idxs = idxs[mask]
    E = E[mask]
    rel = rel[mask]

    # MMR for diversity
    pick_local = _mmr_select(rel, E, k=k)

    out = []
    # used_primary = set()
    for pi in pick_local:
        gi = int(idxs[pi])
        r = res_index.items[gi]

        # pt = _primary_tag(r)
        # simple topical spread (disabled for now)
        # if pt and pt in used_primary:
        #     continue
        # used_primary.add(pt)
        out.append({
            "id": r["id"],
            "title": r["title"],
            "url": r.get("url"),
            "tags": r.get("tags", []),
            "why": _why_for(r, user_profile, user_query),
            "score": round(float(rel[pi]), 3)
        })
        if len(out) >= k:
            break

    return out

# -----------------------------
# --- PROFILE -----------------
# -----------------------------

def is_empty_profile(p: Optional[Dict]) -> bool:
    # Treat None, {} or profiles without any history/seen items as "empty"
    return (not p) or (not p.get("query_history") and not p.get("seen_resource_ids"))

def create_profile(name: Optional[str] = None,
                   locale: str = "en") -> Dict:
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return {
        "user_id": f"u_{uuid.uuid4().hex[:6]}",
        "name": name or "Anonymous",
        "locale": locale,
        "created_at": now,
        "last_active": now,
        "query_history": [],
        "seen_resource_ids": [],
        "clicked_resource_ids": [],
        "top_tags": [],
    }

def profile_update_after_recs(user_profile: Dict, recommendations: List[Dict], user_query: str) -> Dict:
    """
    Returns a minimal mutation you can persist after responding.
    """
    seen = set(user_profile.get("seen_resource_ids") or [])
    for r in recommendations:
        seen.add(r["id"])
    return {
        "append_query_history": {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "text": user_query, "tags": []},
        "seen_resource_ids": sorted(seen),
    }