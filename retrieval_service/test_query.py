import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from retrieval_service.engine.rag_engine import RAGEngine
import chromadb

client = chromadb.HttpClient(host="chromadb", port=8000)

blob_storage = "./data/blob_storage"
docs_path = "./data/documents"

rag = RAGEngine(
    chroma_client=client,
    blob_storage_path=blob_storage,
    documents_path=docs_path
)

print("Engine initialized")

try:
    res = rag.query(prompt="test query", k_text=6, k_image=4, include_encoded_images=False)
    print("Query successful!")
    print("Result paths:", res["paths"])
except Exception as e:
    print(f"Exception: {e}")
