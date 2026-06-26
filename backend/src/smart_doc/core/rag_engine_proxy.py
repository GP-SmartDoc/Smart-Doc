import requests
import os

RETRIEVAL_SERVICE_URL = os.environ.get("RETRIEVAL_SERVICE_URL", "http://retrieval-api:8000")

class RAGEngineProxy:
    def __init__(self, *args, **kwargs):
        pass

    def add_file(self, file_path: str):
        # We assume the file is on a shared volume accessible by both services at the same path
        response = requests.post(f"{RETRIEVAL_SERVICE_URL}/ingest", json={"file_path": file_path})
        response.raise_for_status()
        return response.json()

    def query(self, prompt, k_text=6, k_image=4, document=None, include_encoded_images=True):
        payload = {
            "prompt": prompt,
            "k_text": k_text,
            "k_image": k_image,
            "document": document, 
            "include_encoded_images": include_encoded_images
        }
        response = requests.post(f"{RETRIEVAL_SERVICE_URL}/query", json=payload)
        response.raise_for_status()
        return response.json()

    def list_documents(self):
        response = requests.get(f"{RETRIEVAL_SERVICE_URL}/documents")
        response.raise_for_status()
        return response.json().get("documents", [])
