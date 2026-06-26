import requests
import os
from fastapi import HTTPException

RETRIEVAL_SERVICE_URL = os.environ.get("RETRIEVAL_SERVICE_URL", "http://retrieval-api:8000")

class RAGEngineProxy:
    def __init__(self, *args, **kwargs):
        pass

    def add_file(self, file_path: str):
        # We assume the file is on a shared volume accessible by both services at the same path
        try:
            response = requests.post(f"{RETRIEVAL_SERVICE_URL}/ingest", json={"file_path": file_path})
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise HTTPException(status_code=503, detail=f"Retrieval service unavailable: {e}")

    def query(self, prompt, k_text=6, k_image=4, document=None, include_encoded_images=True):
        payload = {
            "prompt": prompt,
            "k_text": k_text,
            "k_image": k_image,
            "document": document, 
            "include_encoded_images": include_encoded_images
        }
        try:
            response = requests.post(f"{RETRIEVAL_SERVICE_URL}/query", json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise HTTPException(status_code=503, detail=f"Retrieval service unavailable: {e}")

    def list_documents(self):
        try:
            response = requests.get(f"{RETRIEVAL_SERVICE_URL}/documents")
            response.raise_for_status()
            return response.json().get("documents", [])
        except requests.exceptions.RequestException as e:
            raise HTTPException(status_code=503, detail=f"Retrieval service unavailable: {e}")
