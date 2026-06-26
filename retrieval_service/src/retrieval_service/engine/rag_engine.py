import os

import chromadb

from retrieval_service.engine.components import (
    RAGConfig,
    create_collections,
    create_splitters,
    get_torch_device,
    load_caption_model,
    load_yolo_model,
)
from retrieval_service.engine.file_utils import (
    SUPPORTED_DOCUMENT_EXTENSIONS,
    compute_file_hash,
    list_supported_documents,
)
from retrieval_service.engine.image_ingestion import add_image_file, caption_image
from retrieval_service.engine.language import (
    detect_text_language,
    get_text_collection,
    get_text_collection_by_language,
)
from retrieval_service.engine.pdf_ingestion import add_pdf_file
from retrieval_service.engine.query import query_collections
from retrieval_service.engine.spreadsheet_ingestion import add_spreadsheet_file
from retrieval_service.engine.text_ingestion import add_text_file


class RAGEngine:
    def __init__(
        self,
        chroma_client: chromadb.ClientAPI,
        blob_storage_path: str = "backend/data/blob_storage",
        documents_path: str = "backend/data/documents",
        config: RAGConfig | None = None
    ):
        self.__config = config or RAGConfig()
        self.__device = get_torch_device()
        print(f"Using device: {self.__device}")

        self.__blob_storage_path = blob_storage_path
        os.makedirs(self.__blob_storage_path, exist_ok=True)

        self.__documents_path = documents_path
        os.makedirs(self.__documents_path, exist_ok=True)

        self.__collections = create_collections(
            chroma_client,
            self.__device,
            self.__config
        )
        self.__splitters = create_splitters(self.__config)

        # Heavy vision models are loaded only when their ingestion paths need them.
        self.__yolo = None
        self.__caption_processor = None
        self.__caption_model = None

    def _compute_file_hash(self, file_path: str) -> str:
        return compute_file_hash(file_path)

    def _is_file_indexed(self, file_hash: str) -> bool:
        where = {"file_hash": file_hash}
        collections = (
            self.__collections.arabic_text,
            self.__collections.english_text,
            self.__collections.images,
        )

        for collection in collections:
            result = collection.get(where=where, limit=1)
            if result.get("ids"):
                return True

        return False

    def _get_collection(self, text):
        return get_text_collection(
            text,
            self.__collections.arabic_text,
            self.__collections.english_text
        )

    def _get_collection_by_language(self, language):
        return get_text_collection_by_language(
            language,
            self.__collections.arabic_text,
            self.__collections.english_text
        )

    def add_txt(self, file_path):
        file_hash = self._compute_file_hash(file_path)
        if self._is_file_indexed(file_hash):
            return {"status": "skipped", "reason": "duplicate_file"}

        add_text_file(
            file_path,
            self.__splitters.child,
            self._get_collection_by_language,
            detect_text_language,
            file_hash
        )
        return {"status": "indexed"}

    def _caption_image(self, pil_image):
        self._ensure_caption_model()
        return caption_image(
            pil_image,
            self.__caption_processor,
            self.__caption_model
        )

    def _ensure_caption_model(self):
        if self.__caption_processor is not None and self.__caption_model is not None:
            return

        self.__caption_processor, self.__caption_model = load_caption_model(
            self.__config,
            self.__device
        )

    def _ensure_yolo_model(self):
        if self.__yolo is not None:
            return

        # YOLO is only needed for PDF image extraction, not normal querying.
        self.__yolo = load_yolo_model(self.__config, self.__device)

    def add_pdf(self, file_path):
        file_hash = self._compute_file_hash(file_path)
        if self._is_file_indexed(file_hash):
            return {"status": "skipped", "reason": "duplicate_file"}

        self._ensure_yolo_model()
        add_pdf_file(
            file_path,
            self.__documents_path,
            self.__blob_storage_path,
            file_hash,
            self.__splitters.parent,
            self.__splitters.child,
            self.__yolo,
            self.__device,
            self.__config.ignored_layout_classes,
            self._get_collection_by_language,
            detect_text_language,
            self.__collections.images
        )
        return {"status": "indexed"}

    def add_file(self, path: str):
        """
        Add a supported file by detecting its extension and routing it to the
        matching ingestion method.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

        extension = os.path.splitext(path)[1].lower()

        if extension == ".pdf":
            return self.add_pdf(path)

        if extension == ".txt":
            return self.add_txt(path)

        if extension in {".png", ".jpg", ".jpeg", ".webp"}:
            return self.add_image(path)

        if extension in {".xlsx", ".csv"}:
            return self.add_spreadsheet(path)

        supported = ", ".join(sorted(SUPPORTED_DOCUMENT_EXTENSIONS))
        raise ValueError(
            f"Unsupported file type '{extension}'. Supported types: {supported}"
        )

    def add_image(self, file_path):
        file_hash = self._compute_file_hash(file_path)
        if self._is_file_indexed(file_hash):
            return {"status": "skipped", "reason": "duplicate_file"}

        self._ensure_caption_model()
        add_image_file(
            file_path,
            self.__collections.images,
            self._get_collection_by_language,
            detect_text_language,
            self.__caption_processor,
            self.__caption_model,
            file_hash
        )
        return {"status": "indexed"}

    def add_spreadsheet(self, file_path):
        file_hash = self._compute_file_hash(file_path)
        if self._is_file_indexed(file_hash):
            return {"status": "skipped", "reason": "duplicate_file"}

        add_spreadsheet_file(
            file_path,
            self.__splitters.child,
            self._get_collection_by_language,
            detect_text_language,
            file_hash
        )
        return {"status": "indexed"}

    def query(
        self,
        prompt,
        k_text=6,
        k_image=4,
        document=None,
        include_encoded_images=True
    ):
        return query_collections(
            prompt,
            self._get_collection,
            self.__collections.images,
            k_text=k_text,
            k_image=k_image,
            document=document,
            include_encoded_images=include_encoded_images
        )

    def list_documents(self):
        return list_supported_documents(self.__documents_path)
