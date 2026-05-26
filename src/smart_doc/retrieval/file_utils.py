import hashlib
import os


def compute_file_hash(file_path: str) -> str:
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def list_pdf_documents(documents_path: str) -> list[str]:
    if not os.path.exists(documents_path):
        return []

    docs = [
        f for f in os.listdir(documents_path)
        if f.lower().endswith(".pdf")
    ]

    return sorted(docs)
