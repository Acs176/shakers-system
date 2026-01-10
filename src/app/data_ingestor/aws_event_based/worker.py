import os
import time
import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv
load_dotenv()

POLL_INTERVAL_SECONDS = 10
BATCH_SIZE = 5

AWS_REGION = os.environ["AWS_REGION"]
OBJECT_TABLE = os.environ["OBJECT_TABLE"]
BUCKET_NAME = os.environ["BUCKET_NAME"]

dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)
s3 = boto3.client("s3", region_name=AWS_REGION)

table = dynamodb.Table(OBJECT_TABLE)

# ---------- DynamoDB helpers ----------

def find_pending_documents(limit=BATCH_SIZE):
    """
    Find documents that need processing. (or to be deleted)
    NOTE: For simplicity this uses a Scan.
    In production, use a GSI.
    """
    response = table.scan(
        Limit=limit,
        FilterExpression="""
            processing_status = :pending
            OR embedded_version <> latest_version
            OR attribute_not_exists(embedded_version)
            
        """,
        ExpressionAttributeValues={
            ":pending": "PENDING",
        }
    )
    return response.get("Items", [])


def claim_document(doc):
    """
    Atomically claim a document for processing.
    """
    try:
        table.update_item(
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
            print(f"[CLAIMING] Document {doc['object_key']} already claimed. {doc['processing_status']} - {doc['document_status']}")
            return False
        raise


def mark_indexed(doc):
    """
    Mark document as successfully processed.
    """
    table.update_item(
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


def mark_failed(doc, error_message):
    """
    Mark document as failed.
    """
    table.update_item(
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

def cleanup_tombstone(doc):
    print(f"[CLEANUP] Removing vectors for {doc['object_key']}")
    ## vector_store.delete(doc["object_key"])

def process_document(doc):
    """
    Super simple document processing:
    - download from S3
    - print size
    """
    if doc["document_status"] == "TOMBSTONED":
        cleanup_tombstone(doc)
        return
    key = doc["object_key"]
    response = s3.get_object(Bucket=BUCKET_NAME, Key=key)
    data = response["Body"].read()

    print(f"[PROCESS] {key} → {len(data)} bytes")

    # Simulate embedding / indexing
    time.sleep(1)


# ---------- Main worker loop ----------

def run_worker():
    print(f"RAG worker started. Polling every {POLL_INTERVAL_SECONDS} seconds.")
    while True:
        try:
            docs = find_pending_documents()
            if not docs:
                print("No work found.")
            for doc in docs:
                if not claim_document(doc):
                    continue

                try:
                    process_document(doc)
                    mark_indexed(doc)
                    print(f"[DONE] {doc['object_key']}")
                except Exception as e:
                    mark_failed(doc, str(e))
                    print(f"[FAILED] {doc['object_key']} → {e}")

        except Exception as e:
            print(f"[WORKER ERROR] {e}")

        time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    run_worker()
