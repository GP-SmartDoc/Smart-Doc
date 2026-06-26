import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import chromadb

from retrieval_service.engine.rag_engine import RAGEngine
from retrieval_service.worker import ingest_document

app = FastAPI(title="Retrieval Service")

_rag_engine = None

def get_rag_engine():
    global _rag_engine
    if _rag_engine is None:
        host = os.environ.get("CHROMA_HOST", "chromadb")
        port = int(os.environ.get("CHROMA_PORT", 8000))
        client = chromadb.HttpClient(host=host, port=port)
        
        blob_storage = os.environ.get("BLOB_STORAGE_PATH", "/app/data/blob_storage")
        docs_path = os.environ.get("DOCUMENTS_PATH", "/app/data/documents")
        
        _rag_engine = RAGEngine(
            chroma_client=client,
            blob_storage_path=blob_storage,
            documents_path=docs_path
        )
    return _rag_engine


class QueryRequest(BaseModel):
    prompt: str
    k_text: int = 6
    k_image: int = 4
    document: str | None = None
    include_encoded_images: bool = True

class IngestRequest(BaseModel):
    file_path: str

@app.post("/ingest")
def ingest(req: IngestRequest):
    # Enqueue task to Celery
    task = ingest_document.delay(req.file_path)
    return {"task_id": task.id, "status": "queued", "file_path": req.file_path}

@app.post("/query")
def query(req: QueryRequest):
    rag = get_rag_engine()
    result = rag.query(
        prompt=req.prompt,
        k_text=req.k_text,
        k_image=req.k_image,
        document=req.document,
        include_encoded_images=req.include_encoded_images
    )
    return result

from retrieval_service.engine.file_utils import list_supported_documents

@app.get("/documents")
def list_documents():
    docs_path = os.environ.get("DOCUMENTS_PATH", "/app/data/documents")
    return {"documents": list_supported_documents(docs_path)}
