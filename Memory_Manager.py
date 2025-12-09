import os
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# ------------------------
# Configuration
# ------------------------
CHROMA_PERSIST_DIR = "chromadb_storage"  # path to persist your memory
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # sentence-transformers model

# ------------------------
# Initialize Embedding Model
# ------------------------
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# ------------------------
# Initialize Chroma Client
# ------------------------
client = chromadb.Client(
    Settings(
        persist_directory=CHROMA_PERSIST_DIR,
        anonymized_telemetry=False
    )
)

# ------------------------
# Memory Collection
# ------------------------
collection_name = "memory_collection"

collection = client.get_or_create_collection(name=collection_name)

# ------------------------
# Memory Manager Class
# ------------------------
class MemoryManager:
    def __init__(self, collection, embedding_model):
        self.collection = collection
        self.embedding_model = embedding_model

    def add_memory(self, text, metadata=None, id=None):
        """
        Add a text entry to memory.
        """
        embedding = self.embedding_model.encode([text])[0].tolist()
        self.collection.add(
            documents=[text],
            embeddings=[embedding],
            metadatas=[metadata or {}],
            ids=[id or str(os.urandom(8).hex())]
        )

    def query_memory(self, query_text, n_results=3):
        """
        Query memory using semantic similarity.
        """
        embedding = self.embedding_model.encode([query_text])[0].tolist()
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=n_results
        )
        return results

    def clear_memory(self):
        """
        Clear the entire collection.
        """
        self.collection.delete(where={})

# ------------------------
# Example Usage
# ------------------------
if __name__ == "__main__":
    memory = MemoryManager(collection, embedding_model)

    # Add memories
    memory.add_memory("This is my first memory.", {"type": "note"})
    memory.add_memory("Remember to prepare for the graduation project.", {"type": "task"})

    # Query memory
    query = "graduation project"
    results = memory.query_memory(query)
    print("Query results:", results)

    # Optional: clear memory
    # memory.clear_memory()
