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


def list_supported_documents(documents_path: str, user_id: str | None = None) -> list[str]:
    target_path = os.path.join(documents_path, user_id) if user_id else documents_path
    if not os.path.exists(target_path):
        return []

    docs = [
        f for f in os.listdir(target_path)
        if os.path.splitext(f)[1].lower() in SUPPORTED_DOCUMENT_EXTENSIONS
    ]

    return sorted(docs)
