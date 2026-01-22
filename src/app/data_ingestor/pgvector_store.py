from typing import Dict, List, Optional

from sentence_transformers import SentenceTransformer

from src.app.data_ingestor.vector_store import VectorStore

try:
    from sqlalchemy import Column, Integer, MetaData, Table, Text, create_engine, select, text
    from sqlalchemy.dialects.postgresql import JSONB
    from pgvector.sqlalchemy import Vector
except Exception as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "PgVectorStore requires 'pgvector' and a Postgres driver (e.g. 'psycopg')."
    ) from exc


class PgVectorStore(VectorStore):
    def __init__(
        self,
        dsn: str,
        table_name: str = "rag_chunks",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        create_table: bool = True,
    ):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        self.engine = create_engine(dsn)
        self.meta = MetaData()
        self.table = Table(
            table_name,
            self.meta,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("object_key", Text, nullable=False, index=True),
            Column("object_version", Text, nullable=True),
            Column("record", JSONB, nullable=False),
            Column("embedding", Vector(self.dim), nullable=False),
        )
        if create_table:
            with self.engine.begin() as conn:
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                self.meta.create_all(conn)

    def add(self, records: List[Dict]) -> int:
        if not records:
            return 0
        texts = [r["text"] for r in records]
        embs = self.model.encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
        rows = []
        for r, emb in zip(records, embs):
            object_key = r.get("object_key")
            if not object_key:
                raise ValueError("Each record must include object_key for deletion support.")
            rows.append(
                {
                    "object_key": object_key,
                    "object_version": r.get("object_version"),
                    "record": r,
                    "embedding": emb.tolist(),
                }
            )
        with self.engine.begin() as conn:
            conn.execute(self.table.insert(), rows)
        return len(rows)

    def delete_by_object_key(self, object_key: str) -> int:
        with self.engine.begin() as conn:
            result = conn.execute(self.table.delete().where(self.table.c.object_key == object_key))
        return int(result.rowcount or 0)

    def search(self, query: str, k: int = 3) -> List[Dict]:
        q = self.model.encode([query], normalize_embeddings=True)[0]
        distance = self.table.c.embedding.cosine_distance(q)
        stmt = (
            select(
                self.table.c.id,
                self.table.c.record,
                (1 - distance).label("score"),
            )
            .order_by(distance)
            .limit(k)
        )
        with self.engine.begin() as conn:
            rows = conn.execute(stmt).fetchall()
        out: List[Dict] = []
        for row in rows:
            record = dict(row.record or {})
            record["score"] = float(row.score)
            record["vector_id"] = int(row.id)
            out.append(record)
        return out

    def save(self, out_dir: str):
        return None

    @classmethod
    def load(cls, index_dir: str, model_name: Optional[str] = None):
        resolved_model = model_name or "sentence-transformers/all-MiniLM-L6-v2"
        return cls(dsn=index_dir, model_name=resolved_model)
