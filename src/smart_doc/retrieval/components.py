from dataclasses import dataclass, field

import chromadb
import torch
from chromadb.utils import embedding_functions
from chromadb.utils.data_loaders import ImageLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ultralytics import YOLO


@dataclass(frozen=True)
class RAGConfig:
    english_embedding_model: str = "./models/all-MiniLM-L6-v2"
    arabic_embedding_model: str = "./models/GATE-AraBert-v1"
    image_embedding_model: str = "ViT-B-32"
    yolo_model_path: str = "./models/yolo11n_doc_layout.pt"
    parent_chunk_size: int = 1500
    parent_chunk_overlap: int = 200
    child_chunk_size: int = 400
    child_chunk_overlap: int = 50
    ignored_layout_classes: set[str] = field(default_factory=lambda: {
        "Text",
        "Title",
        "Section-header",
        "Page-header",
        "Page-footer",
        "List-item"
    })


@dataclass
class RAGCollections:
    arabic_text: chromadb.Collection
    english_text: chromadb.Collection
    images: chromadb.Collection


@dataclass
class RAGSplitters:
    parent: RecursiveCharacterTextSplitter
    child: RecursiveCharacterTextSplitter


def get_torch_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def create_collections(
    chroma_client: chromadb.ClientAPI,
    device: str,
    config: RAGConfig
) -> RAGCollections:
    english_embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=config.english_embedding_model
    )
    arabic_embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=config.arabic_embedding_model
    )
    image_embedder = embedding_functions.OpenCLIPEmbeddingFunction(
        model_name=config.image_embedding_model,
        device=device
    )

    image_loader = ImageLoader()

    return RAGCollections(
        arabic_text=chroma_client.get_or_create_collection(
            name="arabic_text",
            embedding_function=arabic_embedder
        ),
        english_text=chroma_client.get_or_create_collection(
            name="english_text",
            embedding_function=english_embedder
        ),
        images=chroma_client.get_or_create_collection(
            name="image_collection",
            embedding_function=image_embedder,
            data_loader=image_loader
        )
    )


def create_splitters(config: RAGConfig) -> RAGSplitters:
    return RAGSplitters(
        parent=RecursiveCharacterTextSplitter(
            chunk_size=config.parent_chunk_size,
            chunk_overlap=config.parent_chunk_overlap
        ),
        child=RecursiveCharacterTextSplitter(
            chunk_size=config.child_chunk_size,
            chunk_overlap=config.child_chunk_overlap
        )
    )


def load_yolo_model(config: RAGConfig, device: str):
    yolo = YOLO(config.yolo_model_path)
    yolo.to(device)
    return yolo
