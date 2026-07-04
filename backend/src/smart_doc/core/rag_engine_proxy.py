import requests
import os
from fastapi import HTTPException

RETRIEVAL_SERVICE_URL = os.environ.get("RETRIEVAL_SERVICE_URL", "http://retrieval-api:8000")

class RAGEngineProxy:
    def __init__(self, *args, **kwargs):
        pass

    def add_file(self, file_path: str, user_id: str = None):
        # We assume the file is on a shared volume accessible by both services at the same path
        try:
            payload = {"file_path": file_path}
            if user_id:
                payload["user_id"] = user_id
            response = requests.post(f"{RETRIEVAL_SERVICE_URL}/ingest", json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise HTTPException(status_code=503, detail=f"Retrieval service unavailable: {e}")

    def query(self, prompt, k_text=6, k_image=4, document=None, include_encoded_images=True, user_id=None):
        payload = {
            "prompt": prompt,
            "k_text": k_text,
            "k_image": k_image,
            "document": document, 
            "include_encoded_images": include_encoded_images,
            "user_id": user_id
        }
        try:
            response = requests.post(f"{RETRIEVAL_SERVICE_URL}/query", json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise HTTPException(status_code=503, detail=f"Retrieval service unavailable: {e}")

    def list_documents(self, user_id: str | None = None) -> list[str]:
        try:
            response = requests.get(f"{RETRIEVAL_SERVICE_URL}/documents", params={"user_id": user_id})
            response.raise_for_status()
            return response.json().get("documents", [])
        except requests.exceptions.RequestException as e:
            raise HTTPException(status_code=503, detail=f"Retrieval service unavailable: {e}")
