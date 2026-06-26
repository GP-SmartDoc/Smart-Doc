import os
from celery import Celery
import chromadb

from retrieval_service.engine.rag_engine import RAGEngine

redis_url = os.environ.get("CELERY_BROKER_URL", "redis://redis:6379/0")

celery_app = Celery("retrieval_worker", broker=redis_url, backend=redis_url)

_rag_engine = None

def get_rag_engine():
    global _rag_engine
    if _rag_engine is None:
        host = os.environ.get("CHROMA_HOST", "chromadb")
        port = int(os.environ.get("CHROMA_PORT", 8000))
        # Use HTTP client for standalone chroma
        client = chromadb.HttpClient(host=host, port=port)
        
        blob_storage = os.environ.get("BLOB_STORAGE_PATH", "/app/data/blob_storage")
        docs_path = os.environ.get("DOCUMENTS_PATH", "/app/data/documents")
        
        _rag_engine = RAGEngine(
            chroma_client=client,
            blob_storage_path=blob_storage,
            documents_path=docs_path
        )
    return _rag_engine

@celery_app.task(name="ingest_document")
def ingest_document(file_path: str):
    rag = get_rag_engine()
    print(f"Starting ingestion for {file_path}")
    try:
        result = rag.add_file(file_path)
        print(f"Finished ingestion for {file_path}: {result}")
        return result
    except Exception as e:
        print(f"Error ingesting {file_path}: {e}")
        raise e
