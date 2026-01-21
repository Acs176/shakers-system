import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from src.app.data_ingestor.vector_index import FaissVectorStore
from src.app.data_ingestor.ingestor import md_to_chunks, text_to_chunks, decode_bytes

logger = logging.getLogger(__name__)

DEFAULT_POLL_INTERVAL_SECONDS = 10
DEFAULT_BATCH_SIZE = 5
DEFAULT_PROCESSING_TIMEOUT_SECONDS = 10 * 60  # 10 minutes
DEFAULT_VECTOR_INDEX_PATH = "./rag_index"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass(frozen=True)
class Settings:
    aws_region: str
    object_table: str
    bucket_name: str
    vector_index_path: str
    embedding_model: str
    poll_interval_seconds: int
    batch_size: int
    processing_timeout_seconds: int
    log_level: str

    @staticmethod
    def _require_env(name: str) -> str:
        value = os.getenv(name)
        if not value:
            raise RuntimeError(f"Missing required environment variable: {name}")
        return value

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        return cls(
            aws_region=cls._require_env("AWS_REGION"),
            object_table=cls._require_env("OBJECT_TABLE"),
            bucket_name=cls._require_env("BUCKET_NAME"),
            vector_index_path=os.getenv("VECTOR_INDEX_PATH", DEFAULT_VECTOR_INDEX_PATH),
            embedding_model=os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
            poll_interval_seconds=int(os.getenv("POLL_INTERVAL_SECONDS", DEFAULT_POLL_INTERVAL_SECONDS)),
            batch_size=int(os.getenv("BATCH_SIZE", DEFAULT_BATCH_SIZE)),
            processing_timeout_seconds=int(
                os.getenv("PROCESSING_TIMEOUT_SECONDS", DEFAULT_PROCESSING_TIMEOUT_SECONDS)
            ),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
        )


@dataclass(frozen=True)
class WorkerContext:
    settings: Settings
    table: object
    s3: object
    vector_store: FaissVectorStore


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

# ---------- DynamoDB helpers ----------

def find_pending_documents(ctx: WorkerContext, limit: Optional[int] = None):
    """
    Find pending documents.
    NOTE: For simplicity this uses a Scan. In production, use a GSI.
    Assumes upstream updates set processing_status to PENDING on version changes.
    """
    batch_limit = limit or ctx.settings.batch_size
    response = ctx.table.scan(
        Limit=batch_limit,
        FilterExpression="""
            processing_status = :pending
        """,
        ExpressionAttributeValues={
            ":pending": "PENDING",
        }
    )
    return response.get("Items", [])


def claim_document(ctx: WorkerContext, doc):
    """
    Atomically claim a document for processing.
    """
    try:
        ctx.table.update_item(
            Key={"object_key": doc["object_key"]},
            UpdateExpression="SET processing_status = :processing",
            ConditionExpression="processing_status = :pending",
            ExpressionAttributeValues={
                ":processing": "PROCESSING",
                ":pending": "PENDING",
            },
        )
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] == "ConditionalCheckFailedException":
            logger.info(
                "Document already claimed: key=%s status=%s doc_status=%s",
                doc.get("object_key"),
                doc.get("processing_status"),
                doc.get("document_status"),
            )
            return False
        raise


def mark_indexed(ctx: WorkerContext, doc):
    """
    Mark document as successfully processed.
    """
    ctx.table.update_item(
        Key={"object_key": doc["object_key"]},
        UpdateExpression="""
            SET
                embedded_version = :v,
                processing_status = :indexed,
                last_processed_at = :t
        """,
        ExpressionAttributeValues={
            ":v": doc["latest_version"],
            ":indexed": "INDEXED",
            ":t": int(time.time()),
        },
    )


def mark_skipped(ctx: WorkerContext, doc, reason: str):
    """
    Mark document as skipped (non-indexable).
    """
    ctx.table.update_item(
        Key={"object_key": doc["object_key"]},
        UpdateExpression="""
            SET
                embedded_version = :v,
                processing_status = :skipped,
                skip_reason = :r,
                last_processed_at = :t
        """,
        ExpressionAttributeValues={
            ":v": doc["latest_version"],
            ":skipped": "SKIPPED",
            ":r": reason[:500],
            ":t": int(time.time()),
        },
    )


def mark_failed(ctx: WorkerContext, doc, error_message):
    """
    Mark document as failed.
    """
    ctx.table.update_item(
        Key={"object_key": doc["object_key"]},
        UpdateExpression="""
            SET
                processing_status = :failed,
                error_message = :e,
                last_processed_at = :t
        """,
        ExpressionAttributeValues={
            ":failed": "FAILED",
            ":e": error_message[:500],
            ":t": int(time.time()),
        },
    )


# ---------- Processing logic ----------

def cleanup_tombstone(ctx: WorkerContext, doc):
    logger.info("Removing vectors for %s", doc.get("object_key"))
    removed = ctx.vector_store.delete_by_object_key(doc["object_key"])
    if removed:
        ctx.vector_store.save(ctx.settings.vector_index_path)

def process_document(ctx: WorkerContext, doc) -> Optional[str]:
    """
    Download from S3, normalize text, chunk, and update vector store.
    Returns a skip reason if the document is not indexable.
    """
    if doc["document_status"] == "TOMBSTONED":
        cleanup_tombstone(ctx, doc)
        return None
    ## fetch from s3
    key = doc["object_key"]
    response = ctx.s3.get_object(Bucket=ctx.settings.bucket_name, Key=key)

    ## check filetype
    content_type = response.get("ContentType")
    ext = os.path.splitext(key.lower())[1]
    supported_exts = {".md", ".txt"}
    supported_types = {"text/plain", "text/markdown"}
    is_supported = ext in supported_exts or (content_type in supported_types)
    if not is_supported:
        reason = f"unsupported type ext={ext or 'n/a'} content_type={content_type or 'n/a'}"
        logger.warning("Skipping unsupported type: key=%s %s", key, reason)
        return reason
    ## process supported file
    data = response["Body"].read()
    text = decode_bytes(data, content_type)

    logger.info("Processing key=%s bytes=%s", key, len(data))

    if ext == ".md" or (content_type and "markdown" in content_type):
        records = md_to_chunks(text, source_name=key)
    else:
        records = text_to_chunks(text, source_name=key)
    for r in records:
        r["object_key"] = key
        r["object_version"] = doc.get("latest_version")
    ctx.vector_store.delete_by_object_key(key)
    ctx.vector_store.add(records)
    ctx.vector_store.save(ctx.settings.vector_index_path)
    return None


def recover_stuck_documents(ctx: WorkerContext, limit: Optional[int] = None):
    cutoff = int(time.time()) - ctx.settings.processing_timeout_seconds
    ## object that started being processed before the cutoff or that failed
    response = ctx.table.scan(
        FilterExpression="""
            (processing_status = :p
            AND last_processed_at < :cutoff)
            OR processing_status = :f
        """,
        ExpressionAttributeValues={
            ":p": "PROCESSING",
            ":cutoff": cutoff,
            ":f": "FAILED",
        },
        Limit=limit or ctx.settings.batch_size,
    )
    stuck = response.get("Items", [])
    logger.info("Recover found %s stuck objects", len(stuck))
    for doc in stuck:
        # set them as pending again so they are re-tried
        ctx.table.update_item(
            Key={"object_key": doc["object_key"]},
            UpdateExpression="SET processing_status = :pending",
            ExpressionAttributeValues={":pending": "PENDING"},
        )

# ---------- Main worker loop ----------

def run_worker():
    settings = Settings.from_env()
    configure_logging(settings.log_level)

    dynamodb = boto3.resource("dynamodb", region_name=settings.aws_region)
    s3 = boto3.client("s3", region_name=settings.aws_region)
    table = dynamodb.Table(settings.object_table)
    try:
        vector_store = FaissVectorStore.load(
            settings.vector_index_path,
            model_name=settings.embedding_model,
        )
    except FileNotFoundError:
        vector_store = FaissVectorStore(model_name=settings.embedding_model)
        vector_store.save(settings.vector_index_path)

    ctx = WorkerContext(settings=settings, table=table, s3=s3, vector_store=vector_store)

    logger.info(
        "RAG worker started. poll_interval=%ss batch_size=%s",
        settings.poll_interval_seconds,
        settings.batch_size,
    )
    while True:
        try:
            recover_stuck_documents(ctx)

            docs = find_pending_documents(ctx)
            if not docs:
                logger.info("No work found.")
            for doc in docs:
                if not claim_document(ctx, doc):
                    continue

                try:
                    skip_reason = process_document(ctx, doc)
                    if skip_reason:
                        mark_skipped(ctx, doc, skip_reason)
                        logger.info("Skipped key=%s", doc.get("object_key"))
                    else:
                        mark_indexed(ctx, doc)
                        logger.info("Done key=%s", doc.get("object_key"))
                except Exception as e:
                    mark_failed(ctx, doc, str(e))
                    logger.exception("Failed key=%s", doc.get("object_key"))

        except Exception as e:
            logger.exception("Worker error")

        time.sleep(settings.poll_interval_seconds)


if __name__ == "__main__":
    run_worker()
