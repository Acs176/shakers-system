import hashlib
import json
import pathlib
from typing import Dict, Iterable, List, Optional

import faiss
import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer

from src.app.data_ingestor.ingestor import read_markdown, md_to_chunks
from src.app.data_ingestor.vector_store import VectorStore


class FaissVectorStore(VectorStore):
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexIDMap2(faiss.IndexFlatIP(self.dim))
        self.meta: Dict[int, Dict] = {}
        self.object_map: Dict[str, List[int]] = {}

    @staticmethod
    def _vector_id(seed: str) -> int:
        digest = hashlib.sha1(seed.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], byteorder="big", signed=False)

    @staticmethod
    def _ensure_np(vectors: Iterable) -> np.ndarray:
        if not isinstance(vectors, np.ndarray):
            vectors = np.array(vectors, dtype="float32")
        return vectors.astype("float32")

    def _index_ids(self, records: List[Dict]) -> np.ndarray:
        ids: List[int] = []
        for r in records:
            object_key = r.get("object_key")
            if not object_key:
                raise ValueError("Each record must include object_key for deletion support.")
            chunk_id = r.get("id") or r.get("chunk_id")
            if not chunk_id:
                raise ValueError("Each record must include id or chunk_id.")
            version = r.get("object_version", "")
            vec_id = self._vector_id(f"{object_key}:{version}:{chunk_id}")
            r["vector_id"] = vec_id
            ids.append(vec_id)
        return np.array(ids, dtype="int64")

    def add(self, records: List[Dict]) -> int:
        if not records:
            return 0
        texts = [r["text"] for r in records]
        embs = self.model.encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
        embs = self._ensure_np(embs)
        ids = self._index_ids(records)
        self.index.add_with_ids(embs, ids)
        for r in records:
            vec_id = r["vector_id"]
            self.meta[vec_id] = r
            self.object_map.setdefault(r["object_key"], []).append(vec_id)
        return len(records)

    def delete_by_object_key(self, object_key: str) -> int:
        ids = self.object_map.get(object_key, [])
        if not ids:
            return 0
        if not hasattr(self.index, "remove_ids"):
            logger.warning("index.delete.unsupported", object_key=object_key)
            return 0
        id_arr = np.array(ids, dtype="int64")
        self.index.remove_ids(id_arr)
        for vec_id in ids:
            self.meta.pop(vec_id, None)
        self.object_map.pop(object_key, None)
        return len(ids)

    def search(self, query: str, k: int = 3) -> List[Dict]:
        if self.index.ntotal == 0:
            return []
        q = self.model.encode([query], normalize_embeddings=True)
        q = self._ensure_np(q)
        D, I = self.index.search(q, k)
        sims = D[0].tolist()
        ids = I[0].tolist()
        out: List[Dict] = []
        for score, vec_id in zip(sims, ids):
            if vec_id == -1:
                continue
            meta = self.meta.get(vec_id)
            if not meta:
                continue
            d = meta.copy()
            d["score"] = float(score)
            out.append(d)
        return out

    def save(self, out_dir: str):
        out = pathlib.Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(out / "vectors.faiss"))
        payload = {
            "model_name": self.model_name,
            "records": list(self.meta.values()),
        }
        (out / "meta.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, index_dir: str, model_name: Optional[str] = None):
        idx_path = pathlib.Path(index_dir) / "vectors.faiss"
        meta_path = pathlib.Path(index_dir) / "meta.json"
        if not idx_path.exists() or not meta_path.exists():
            raise FileNotFoundError(f"Missing index files in {index_dir}")
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            records = payload
            resolved_model = model_name or "sentence-transformers/all-MiniLM-L6-v2"
        else:
            records = payload.get("records", [])
            resolved_model = model_name or payload.get("model_name") or "sentence-transformers/all-MiniLM-L6-v2"
        obj = cls(model_name=resolved_model)
        obj.index = faiss.read_index(str(idx_path))
        for idx, r in enumerate(records):
            vec_id = r.get("vector_id")
            if vec_id is None:
                vec_id = idx
                r["vector_id"] = vec_id
            vec_id = int(vec_id)
            if not r.get("object_key"):
                r["object_key"] = r.get("source", "unknown")
            obj.meta[vec_id] = r
            obj.object_map.setdefault(r["object_key"], []).append(vec_id)
        return obj


def build_index(
    kb_dir: str,
    out_dir: str,
    chunk_chars: int = 1200,
    overlap: int = 200,
    model: str = "sentence-transformers/all-MiniLM-L6-v2",
):
    kb = pathlib.Path(kb_dir)
    files = sorted([p for p in kb.rglob("*.md") if p.is_file()])
    if not files:
        raise SystemExit(f"No markdown files found in {kb_dir}")

    vx = FaissVectorStore(model_name=model)
    all_records: List[Dict] = []
    for p in files:
        md = read_markdown(p)
        recs = md_to_chunks(md, source_name=p.name, chunk_chars=chunk_chars, overlap=overlap)
        for r in recs:
            r["object_key"] = p.name
            r["object_version"] = "local"
        all_records.extend(recs)
    vx.add(all_records)
    vx.save(out_dir)
    logger.info("index.end", summary=f"Indexed {len(files)} files → {len(all_records)} chunks → {out_dir}")
