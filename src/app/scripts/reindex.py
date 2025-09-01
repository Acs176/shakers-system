import argparse
from loguru import logger
from src.app.data_ingestor.vector_index import reindex, VectorIndex

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a knowledge base index.")
    parser.add_argument("kb_dir", help="Path to the knowledge base directory")
    parser.add_argument("index_dir", help="Path to the existing index")

    args = parser.parse_args()

    vx = VectorIndex.load(args.index_dir)
    result = reindex(vx, args.kb_dir)
    logger.info(result)

