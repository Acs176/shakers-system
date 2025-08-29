import pathlib, json
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import numpy as np
import faiss
from loguru import logger
from src.app.data_ingestor.ingestor import read_markdown, md_to_chunks


class VectorIndex:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatIP(self.dim)  # inner product -> cosine with normalized vectors
        self.meta: List[Dict] = []
        self.count = 0

    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
        return v / n

    def add(self, records: List[Dict]):
        texts = [r["text"] for r in records]
        embs = self.model.encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
        if not isinstance(embs, np.ndarray):
            embs = np.array(embs, dtype="float32")
        embs = embs.astype("float32")
        self.index.add(embs)
        self.meta.extend(records)
        self.count += len(records)

    def save(self, out_dir: str):
        out = pathlib.Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(out / "vectors.faiss"))
        (out / "meta.json").write_text(json.dumps(self.meta, ensure_ascii=False, indent=2), encoding="utf-8")
        # (out / "model.txt").write_text(self.model.model_name, encoding="utf-8")

    @classmethod
    def load(cls, index_dir: str, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        obj = cls(model_name=model_name)
        idx_path = pathlib.Path(index_dir) / "vectors.faiss"
        meta_path = pathlib.Path(index_dir) / "meta.json"
        if not idx_path.exists() or not meta_path.exists():
            raise FileNotFoundError(f"Missing index files in {index_dir}")
        obj.index = faiss.read_index(str(idx_path))
        obj.meta = json.loads(meta_path.read_text(encoding="utf-8"))
        obj.count = len(obj.meta)
        return obj
    
def build_index(kb_dir: str, out_dir: str, chunk_chars=1200, overlap=200, model="sentence-transformers/all-MiniLM-L6-v2"):
    kb = pathlib.Path(kb_dir)
    files = sorted([p for p in kb.rglob("*.md") if p.is_file()])
    if not files:
        raise SystemExit(f"No markdown files found in {kb_dir}")

    vx = VectorIndex(model_name=model)
    all_records = []
    for p in files:
        md = read_markdown(p)
        recs = md_to_chunks(md, source_name=p.name, chunk_chars=chunk_chars, overlap=overlap)
        all_records.extend(recs)
    vx.add(all_records)
    vx.save(out_dir)
    logger.info("index.end", summary=f"Indexed {len(files)} files → {vx.count} chunks → {out_dir}")