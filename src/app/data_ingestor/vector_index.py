import pathlib, json
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import numpy as np
import faiss
from loguru import logger
from src.app.data_ingestor.ingestor import read_markdown, md_to_chunks, content_hash

class VectorIndex:
    model_name: str
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
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
        (out / "model.txt").write_text(self.model_name, encoding="utf-8")

    def fetch(self, *, source: str) -> list[dict]:
        return [rec for rec in self.meta if rec.get("source") == source]
    
    def _rebuild(self):
        # Re-encode everything in self.meta and rebuild FAISS
        texts = [r["text"] for r in self.meta]
        if texts:
            embs = self.model.encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
            embs = np.asarray(embs, dtype="float32")
            self.index = faiss.IndexFlatIP(self.dim)
            self.index.add(embs)
        else:
            self.index = faiss.IndexFlatIP(self.dim)
        self.count = len(self.meta)

    def upsert(self, records: list[dict]):
        if not records:
            return
        # replace any existing by id; insert if new
        by_id = {r["id"]: r for r in self.meta}  # current meta as a map
        for r in records:
            by_id[r["id"]] = r
        self.meta = list(by_id.values())
        self._rebuild()
    
    def delete(self, ids: list[str]):
        if not ids:
            return
        ids_set = set(ids)
        # drop from meta and rebuild
        self.meta = [r for r in self.meta if r.get("id") not in ids_set]
        self._rebuild()


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

def reindex_doc(vx: VectorIndex, source_path: str, chunk_chars=1200, overlap=200):
    md_text = read_markdown(source_path)
    new_recs = md_to_chunks(md_text, source_name=source_path, chunk_chars=chunk_chars, overlap=overlap)

    old_recs = vx.fetch(source=source_path)
    old_map = {r["id"]: r for r in old_recs}
    new_map = {r["id"]: r for r in new_recs}

    # upserts = new or changed (by hash)
    upserts = [r for cid, r in new_map.items() if cid not in old_map or old_map[cid]["hash"] != r["hash"]]
    # deletes = old ids that no longer exist
    deletes = [cid for cid in old_map.keys() if cid not in new_map]

    if upserts: vx.upsert(upserts)
    if deletes: vx.delete(deletes)

    return {"upserts": len(upserts), "deletes": len(deletes), "skipped": len(new_map) - len(upserts)}

def reindex(vx: VectorIndex, kb_dir: str, **kwargs):
    kb = pathlib.Path(kb_dir)
    files = sorted(p for p in kb.rglob("*.md") if p.is_file())
    stats = []
    for p in files:
        res = reindex_doc(vx, source_path=p, **kwargs)
        stats.append(res)
    return stats

def reindex_doclevel(kb_dir: str, index_dir: str, chunk_chars=1200, overlap=200, model="sentence-transformers/all-MiniLM-L6-v2"):
    kb = pathlib.Path(kb_dir)
    idx = pathlib.Path(index_dir)
    idx.mkdir(parents=True, exist_ok=True)

    model_path = idx / "model.txt"
    model_path = pathlib.Path(model_path)
    old_model = model_path.read_text(encoding="utf-8").strip() if model_path.exists() else None

    vx = VectorIndex.load(idx) if idx.exists() else VectorIndex(model_name=model)
    if old_model != model:
        #rebuild full if model changed
        build_index(kb_dir, index_dir, chunk_chars, overlap, model)
        return

    files = sorted(p for p in kb.rglob("*.md") if p.is_file())
    seen: set[str] = set()

    for p in files:
        doc_id = str(p.relative_to(kb))
        seen.add(doc_id)

        md = read_markdown(p)
        recs = md_to_chunks(md, source_name=doc_id, chunk_chars=chunk_chars, overlap=overlap)

        # doc hash from normalized chunk text
        doc_text = "\n\n".join(_norm(r["text"]) for r in recs)
        new_hash = _sha(doc_text)

        prev = manifest["docs"].get(doc_id)
        if prev and prev["doc_hash"] == new_hash:
            continue  # unchanged → skip everything

        # delete old chunks for this doc (if any), then add new ones
        if prev:
            vx.delete(ids=prev["chunk_ids"])
        vx.add(recs)

        manifest["docs"][doc_id] = {
            "doc_hash": new_hash,
            "chunk_ids": [r["id"] for r in recs],
        }

    # remove docs that were deleted from disk
    removed = [d for d in list(manifest["docs"].keys()) if d not in seen]
    for d in removed:
        vx.delete(ids=manifest["docs"][d]["chunk_ids"])
        manifest["docs"].pop(d, None)

    vx.save(idx)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))