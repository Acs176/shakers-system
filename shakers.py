#!/usr/bin/env python3
"""
Minimal RAG: Indexer + Retriever + Generator (with citations)

Usage:
  Index: python rag_minimal.py index --kb ./knowledge_base --out ./rag_index
  Ask:   python rag_minimal.py ask --index ./rag_index --q "How do payments work?"

Env (optional):
  LLM_PROVIDER = gemini | openai | groq | none
  GEMINI_API_KEY / OPENAI_API_KEY / GROQ_API_KEY
"""

import os, re, json, time, argparse, math, pathlib, textwrap
from typing import List, Dict, Tuple, Optional
from dotenv import load_dotenv

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# -----------------------------
# --------- CHUNKING ----------
# -----------------------------
H1 = re.compile(r'^\s*#\s+(.*)$', re.MULTILINE)
H2_SPLIT = re.compile(r'^\s*##\s+(.*)$', re.MULTILINE)

def read_markdown(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")

def extract_title(md_text: str, fallback: str) -> str:
    m = H1.search(md_text)
    return (m.group(1).strip() if m else fallback).strip()

def section_aware_splits(md_text: str) -> List[Tuple[str, str]]:
    """
    Returns list of (section_title or '', section_text). Includes preface as ''.
    """
    sections = []
    last_end = 0
    last_title = ""
    for m in H2_SPLIT.finditer(md_text):
        sec_title = m.group(1).strip()
        sec_start = m.start()
        # previous block
        if sec_start > last_end:
            block = md_text[last_end:sec_start].strip()
            if block:
                sections.append((last_title, block))
        last_title = sec_title
        last_end = m.end()
    # tail
    tail = md_text[last_end:].strip()
    if tail:
        sections.append((last_title, tail))
    if not sections:
        sections = [("", md_text)]
    return sections

def normalize_ws(s: str) -> str:
    s = re.sub(r'[ \t]+', ' ', s)
    s = re.sub(r'\n{3,}', '\n\n', s)
    return s.strip()

def chunk_text(text: str, chunk_chars: int = 1200, overlap: int = 200) -> List[str]:
    """
    Simple char-based chunking with overlap; keeps lists/paragraphs intact where possible.
    """
    text = normalize_ws(text)
    if len(text) <= chunk_chars:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_chars)
        # Try to break at a newline or sentence end for better readability
        window = text[start:end]
        cut = max(window.rfind("\n"), window.rfind(". "))
        if cut == -1 or cut < int(chunk_chars * 0.5):
            cut = len(window)
        chunk = window[:cut].strip()
        if chunk:
            chunks.append(chunk)
        start = max(0, start + chunk_chars - overlap)
        if start >= len(text):
            break
    # ensure tail
    if chunks and chunks[-1] != text[-len(chunks[-1]):]:
        tail_start = chunks[-1] and text.rfind(chunks[-1]) + len(chunks[-1]) or 0
        if tail_start < len(text):
            tail = text[tail_start:].strip()
            if tail:
                chunks.append(tail[:chunk_chars])
    # dedupe tiny fragments
    chunks = [c for c in chunks if len(c) > 100]
    return chunks or [text]

def md_to_chunks(md_text: str, source_name: str, chunk_chars=1200, overlap=200) -> List[Dict]:
    title = extract_title(md_text, fallback=pathlib.Path(source_name).stem)
    sections = section_aware_splits(md_text)
    items = []
    idx = 0
    for sec_title, sec_text in sections:
        for part in chunk_text(sec_text, chunk_chars=chunk_chars, overlap=overlap):
            idx += 1
            items.append({
                "id": f"{source_name}#chunk_{idx:03d}",
                "title": title,
                "section": sec_title or "Summary",
                "source": source_name,
                "text": part
            })
    return items

# -----------------------------
# --------- INDEXING ----------
# -----------------------------
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

# -----------------------------
# --------- RETRIEVAL ---------
# -----------------------------
def top_k_search(vx: VectorIndex, query: str, k: int = 4) -> List[Dict]:
    q = vx.model.encode([query], normalize_embeddings=True)
    if not isinstance(q, np.ndarray):
        q = np.array(q, dtype="float32")
    q = q.astype("float32")
    D, I = vx.index.search(q, k)  # inner product similarity
    sims = D[0].tolist()
    idxs = I[0].tolist()
    out = []
    for score, i in zip(sims, idxs):
        if i == -1:
            continue
        d = vx.meta[i].copy()
        d["score"] = float(score)  # in [-1,1]
        out.append(d)
    return out

# -----------------------------
# --------- GENERATION --------
# -----------------------------
def build_prompt(query: str, docs: List[Dict]) -> str:
    parts = []
    for d in docs:
        parts.append(
            f"{d['text']}\nSOURCE: {d['title']} — {d['section']} — {d['source']} — {d['id']}\n"
        )
    context = "\n\n".join(parts)
    system = (
        "You are a helpful support assistant for the Shakers platform.\n"
        "Answer ONLY using the provided context. If the answer is not present, say you don't have that information.\n"
        "Be concise and include a 'Sources:' list referencing the SOURCE lines.\n"
    )
    return f"{system}\nContext:\n{context}\n\nUser question:\n{query}\n\nAnswer:"

def try_gemini(prompt: str) -> Optional[str]:
    try:
        import google.generativeai as genai
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            return None
        genai.configure(api_key=key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        resp = model.generate_content(prompt)
        return resp.text.strip()
    except Exception:
        return None

def extractive_fallback(query: str, docs: List[Dict]) -> str:
    """
    Simple extractive answer: return a concise summary (first 3-5 sentences)
    from the top doc(s) + a Sources list. Works without an LLM.
    """
    joined = " ".join(d["text"] for d in docs)
    sentences = re.split(r'(?<=[\.\!\?])\s+', joined)
    summary = " ".join(sentences[:5]).strip()
    lines = []
    lines.append(summary if summary else "I don't have that information in the provided context.")
    lines.append("\nSources:")
    for d in docs:
        lines.append(f"- {d['title']} — {d['section']} ({d['source']}#{d['id'].split('#')[-1]})")
    return "\n".join(lines).strip()

def generate_answer(query: str, docs: List[Dict]) -> str:
    prompt = build_prompt(query, docs)
    provider = (os.getenv("LLM_PROVIDER") or "").lower()
    text = None
    if provider in ("gemini", ""):
        text = try_gemini(prompt)
    if text is None:
        text = extractive_fallback(query, docs)
    return text

# -----------------------------
# --------- RAG ORCHESTRATION -
# -----------------------------
def ask(index_dir: str, query: str, k: int = 4, oos_threshold: float = 0.22) -> Dict:
    t0 = time.perf_counter()
    vx = VectorIndex.load(index_dir)
    hits = top_k_search(vx, query, k=k)

    out_of_scope = False
    if not hits:
        out_of_scope = True
    else:
        max_sim = max(h["score"] for h in hits)
        # Convert inner product (cosine) [-1,1] to [0,1] if you want a human-friendly score
        max_sim01 = (max_sim + 1.0) / 2.0
        out_of_scope = max_sim01 < oos_threshold

    if out_of_scope:
        answer = "I don't have information on this in the current knowledge base."
        citations = []
    else:
        answer = generate_answer(query, hits)
        citations = [{
            "title": h["title"],
            "section": h["section"],
            "source": h["source"],
            "chunk_id": h["id"],
            "score": round((h["score"] + 1.0) / 2.0, 3)  # normalized 0..1
        } for h in hits]

    latency_ms = int((time.perf_counter() - t0) * 1000)
    return {
        "answer": answer,
        "citations": citations,
        "out_of_scope": out_of_scope,
        "latency_ms": latency_ms
    }

# -----------------------------
# --------- BUILD INDEX CLI ---
# -----------------------------
def build_index(kb_dir: str, out_dir: str, chunk_chars=1200, overlap=200, model="sentence-transformers/all-MiniLM-L6-v2"):
    kb = pathlib.Path(kb_dir)
    files = sorted([p for p in kb.glob("*.md") if p.is_file()])
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
    print(f"Indexed {len(files)} files → {vx.count} chunks → {out_dir}")

# -----------------------------
# --------------- MAIN --------
# -----------------------------
def main():
    load_dotenv()
    ap = argparse.ArgumentParser(description="Minimal RAG (Indexer + Retriever + Generator)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_index = sub.add_parser("index", help="Index .md files from a folder")
    ap_index.add_argument("--kb", required=True, help="Folder with .md docs")
    ap_index.add_argument("--out", required=True, help="Output folder for the FAISS index")
    ap_index.add_argument("--chunk_chars", type=int, default=1200)
    ap_index.add_argument("--overlap", type=int, default=200)
    ap_index.add_argument("--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")

    ap_ask = sub.add_parser("ask", help="Query the index with a question")
    ap_ask.add_argument("--index", required=True, help="Folder with vectors.faiss + meta.json")
    ap_ask.add_argument("--q", required=True, help="User query")
    ap_ask.add_argument("--k", type=int, default=4)
    ap_ask.add_argument("--oos_threshold", type=float, default=0.22)

    args = ap.parse_args()

    if args.cmd == "index":
        build_index(args.kb, args.out, chunk_chars=args.chunk_chars, overlap=args.overlap, model=args.model)
    elif args.cmd == "ask":
        res = ask(args.index, args.q, k=args.k, oos_threshold=args.oos_threshold)
        print(json.dumps(res, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
