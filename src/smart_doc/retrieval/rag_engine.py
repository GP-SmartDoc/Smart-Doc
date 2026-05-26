import os

import chromadb

from smart_doc.retrieval.components import (
    RAGConfig,
    create_collections,
    create_splitters,
    get_torch_device,
    load_caption_model,
    load_yolo_model,
)
from smart_doc.retrieval.file_utils import compute_file_hash, list_pdf_documents
from smart_doc.retrieval.image_ingestion import add_image_file, caption_image
from smart_doc.retrieval.language import detect_text_language, get_text_collection
from smart_doc.retrieval.pdf_ingestion import add_pdf_file
from smart_doc.retrieval.query import query_collections
from smart_doc.retrieval.text_ingestion import add_text_file


class RAGEngine:
    def __init__(
        self,
        chroma_client: chromadb.ClientAPI,
        blob_storage_path: str = "./data/blob_storage",
        documents_path: str = "./data/documents",
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
        self.__yolo = load_yolo_model(self.__config, self.__device)
        self.__caption_processor = None
        self.__caption_model = None

    def _compute_file_hash(self, file_path: str) -> str:
        return compute_file_hash(file_path)

    def _get_collection(self, text):
        return get_text_collection(
            text,
            self.__collections.arabic_text,
            self.__collections.english_text
        )

    def add_txt(self, file_path):
        file_hash = self._compute_file_hash(file_path)
        add_text_file(
            file_path,
            self.__splitters.child,
            self._get_collection,
            detect_text_language,
            file_hash
        )

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

    def add_pdf(self, file_path):
        file_hash = self._compute_file_hash(file_path)
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
            self._get_collection,
            detect_text_language,
            self.__collections.images
        )

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

        supported = ".pdf, .txt, .png, .jpg, .jpeg, .webp"
        raise ValueError(
            f"Unsupported file type '{extension}'. Supported types: {supported}"
        )

    def add_image(self, file_path):
        self._ensure_caption_model()
        file_hash = self._compute_file_hash(file_path)
        add_image_file(
            file_path,
            self.__collections.images,
            self._get_collection,
            detect_text_language,
            self.__caption_processor,
            self.__caption_model,
            file_hash
        )

    def query(
        self,
        prompt,
        k_text=6,
        k_image=4,
        document=None
    ):
        return query_collections(
            prompt,
            self._get_collection,
            self.__collections.images,
            k_text=k_text,
            k_image=k_image,
            document=document
        )

    def list_documents(self):
        return list_pdf_documents(self.__documents_path)
