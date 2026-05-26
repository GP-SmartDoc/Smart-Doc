import hashlib
import os


SUPPORTED_DOCUMENT_EXTENSIONS = {
    ".pdf",
    ".txt",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".xlsx",
    ".csv",
}


def compute_file_hash(file_path: str) -> str:
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def list_supported_documents(documents_path: str) -> list[str]:
    if not os.path.exists(documents_path):
        return []

    docs = [
        f for f in os.listdir(documents_path)
        if os.path.splitext(f)[1].lower() in SUPPORTED_DOCUMENT_EXTENSIONS
    ]

    return sorted(docs)
