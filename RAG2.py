import chromadb
from chromadb.utils import embedding_functions
from chromadb.utils.data_loaders import ImageLoader
from huggingface_hub import snapshot_download
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PIL import Image
import fitz, io, os, uuid
from ultralytics import YOLO

class RAGEngine:
    """
    GP-ready RAG Engine:
    - Parent–child text splitting
    - YOLO-based region detection (ignores text)
    - Geometry-based filtering only
    - MPNet embeddings for semantic search
    """

    def __init__(self, chroma_client, blob_storage_path="./blob_storage"):
        self.__client = chroma_client
        self.__blob_storage_path = blob_storage_path
        os.makedirs(blob_storage_path, exist_ok=True)

        # ---------------- EMBEDDERS ----------------
        mpnet_dir = "./models/all-mpnet-base-v2"
        self.__text_embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=mpnet_dir
        )

        self.__image_embedder = embedding_functions.OpenCLIPEmbeddingFunction(
            model_name="ViT-B-32",
            device="cpu"  # change to "cuda" if GPU available
        )
        self.__image_loader = ImageLoader()

        # ---------------- COLLECTIONS ----------------
        self.__text_collection = self.__client.get_or_create_collection(
            name="text_collection",
            embedding_function=self.__text_embedder
        )
        self.__image_collection = self.__client.get_or_create_collection(
            name="image_collection",
            embedding_function=self.__image_embedder,
            data_loader=self.__image_loader
        )

        # ---------------- TEXT SPLITTING ----------------
        self.__parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        self.__child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)

        # ---------------- YOLO ----------------
        self.__yolo = YOLO("./models/yolo11n_doc_layout.pt")

        # ---------------- IGNORED CLASSES ----------------
        self.__ignored_layout_classes = {
            "Text", "Title", "Section-header", "Page-header", "Page-footer", "List-item"
        }

    # =========================================================
    # PDF INGESTION
    # =========================================================
    def add_pdf(self, file_path):
        filename = os.path.basename(file_path)
        # Skip if already indexed
       

        loader = PyMuPDFLoader(file_path)
        docs = loader.load()
        pdf = fitz.open(file_path)

        for page_index, doc in enumerate(docs):
            # ---------------- TEXT ----------------
            parent_chunks = self.__parent_splitter.split_text(doc.page_content)
            for p_id, parent in enumerate(parent_chunks):
                child_chunks = self.__child_splitter.split_text(parent)
                for c_id, child in enumerate(child_chunks):
                    self.__text_collection.add(
                        documents=[child],
                        ids=[f"{filename}_p{page_index}_P{p_id}_C{c_id}"],
                        metadatas=[{"page": page_index, "parent_text": parent}]
                    )

            # ---------------- IMAGE ----------------
            page = pdf[page_index]
            pix = page.get_pixmap(dpi=200)
            page_img = Image.open(io.BytesIO(pix.tobytes("png")))
            page_area = page_img.width * page_img.height

            results = self.__yolo(page_img, conf=0.35)
            if not results: 
                continue
            result = results[0]

            for det_id, box in enumerate(result.boxes):
                class_id = int(box.cls[0])
                class_name = result.names[class_id]
                if class_name in self.__ignored_layout_classes:
                    continue

                x0, y0, x1, y1 = map(int, box.xyxy[0].tolist())
                w, h = x1 - x0, y1 - y0
                area_ratio = (w * h) / page_area
                if area_ratio > 0.6 or w < 200 or h < 200:
                    continue
                aspect_ratio = w / h
                if aspect_ratio > 5 or aspect_ratio < 0.2:
                    continue

                crop = page_img.crop((x0, y0, x1, y1))
                img_path = os.path.join(self.__blob_storage_path, f"{filename}_p{page_index}_fig{det_id}.png")
                crop.save(img_path)
                self.add_image(img_path)

        print(f"PDF indexed correctly: {filename}")

    # =========================================================
    # IMAGE INGESTION
    # =========================================================
    def add_image(self, file_path):
        abs_path = os.path.abspath(file_path)
        self.__image_collection.add(
            ids=[str(uuid.uuid4())],
            uris=[abs_path],
            metadatas=[{"source": abs_path}]
        )

    # =========================================================
    # QUERY
    # =========================================================
    def query(self, prompt, k_text=5, k_image=3):
        text_res = self.__text_collection.query(query_texts=[prompt], n_results=k_text)
        img_res = self.__image_collection.query(query_texts=[prompt], n_results=k_image, include=["uris"])
        return {
            "text": text_res.get("documents", [[]])[0],
            "images": img_res.get("uris", [[]])[0]
        }
