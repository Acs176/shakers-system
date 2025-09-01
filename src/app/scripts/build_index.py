import argparse
from src.app.data_ingestor.vector_index import build_index, VectorIndex

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a knowledge base index.")
    parser.add_argument("kb_dir", help="Path to the knowledge base directory")
    parser.add_argument("out_dir", help="Path to the output directory")
    parser.add_argument("--chunk_chars", type=int, default=1200, help="Number of characters per chunk")
    parser.add_argument("--overlap", type=int, default=200, help="Number of overlapping characters between chunks")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2", help="Embedding model to use")

    args = parser.parse_args()

    build_index(
        kb_dir=args.kb_dir,
        out_dir=args.out_dir,
        chunk_chars=args.chunk_chars,
        overlap=args.overlap,
        model=args.model,
    )