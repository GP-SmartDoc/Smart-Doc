import chromadb
from chromadb.utils import embedding_functions

print("Loading model...")
embedder = embedding_functions.OpenCLIPEmbeddingFunction(
    model_name="ViT-B-32",
    device="cpu"
)
try:
    print("Embedding...")
    res = embedder(["high"])
    print("Success:", len(res))
except Exception as e:
    print("Exception:", e)
