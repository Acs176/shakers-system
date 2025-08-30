import argparse

def setup_args():
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
    ap_ask.add_argument("--uid", type=str, default="")
    ap_ask.add_argument("--oos_threshold", type=float, default=0.65)

    args = ap.parse_args()
    return args