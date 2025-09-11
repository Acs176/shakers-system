import numpy as np
from sentence_transformers import SentenceTransformer


_EVAL_MODEL = None
def semantic_similarity(pred: str, gold: str, model_name: str = "sentence-transformers/all-mpnet-base-v2") -> float:
    """
    Cosine similarity between contextual embeddings of predicted and expected answers.
    Uses a separate sentence-transformer model from the retriever to avoid bias.
    Returns a score in [-1, 1], typically [0, 1] after normalization.
    """
    global _EVAL_MODEL
    # Handle trivial cases
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    
    if _EVAL_MODEL is None or getattr(_EVAL_MODEL, "model_name_or_path", None) != model_name:
        _EVAL_MODEL = SentenceTransformer(model_name)
    vecs = _EVAL_MODEL.encode([pred, gold], normalize_embeddings=True)
    if not isinstance(vecs, np.ndarray):
        vecs = np.array(vecs, dtype="float32")
    v1, v2 = vecs[0].astype("float32"), vecs[1].astype("float32")
    return float(np.dot(v1, v2))


