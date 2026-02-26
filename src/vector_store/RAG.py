import chromadb
import fitz
import io
import os
import uuid
import torch  
import hashlib
import shutil

from chromadb.utils import embedding_functions
from chromadb.utils.data_loaders import ImageLoader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PIL import Image
from ultralytics import YOLO
from pathlib import Path
from transformers import BlipProcessor, BlipForConditionalGeneration
from src.utils.image import encode_image_from_path


class RAGEngine:

    def __init__(self, chroma_client: chromadb.ClientAPI):

        self.__client = chroma_client
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.__device = device
        print(f"Using device: {self.__device}")
        self.__blob_storage_path = "./blob_storage"
        os.makedirs(self.__blob_storage_path, exist_ok=True)

        # ---------------- DOCUMENT STORAGE ----------------

        self.__documents_path = "./documents"
        os.makedirs(self.__documents_path, exist_ok=True)

        # ---------------- EMBEDDERS ----------------

        self.__text_embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="./models/all-MiniLM-L6-v2",
            device=self.__device
        )

        self.__image_embedder = embedding_functions.OpenCLIPEmbeddingFunction(
            model_name="ViT-B-32",
            device=self.__device
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


        # ---------------- TEXT SPLITTER ----------------

        self.__parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200
        )

        self.__child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=50
        )


        # ---------------- YOLO ----------------

        self.__yolo = YOLO("./models/yolo11n_doc_layout.pt")
        self.__yolo.to(self.__device)

        # ---------------- IMAGE CAPTIONING ----------------

        # ---------------- IGNORED CLASSES ----------------

        self.__ignored_layout_classes = {

            "Text",
            "Title",
            "Section-header",
            "Page-header",
            "Page-footer",
            "List-item"
        }

    # =====================================================
    # HELPER: Compute MD5 hash of a file
    # =====================================================
    def _compute_file_hash(self, file_path: str) -> str:
        hasher = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    # =====================================================
    # ADD PDF
    # =====================================================

    def add_pdf(self, file_path):

        filename = os.path.basename(file_path)

        # copy pdf into documents folder
        stored_path = os.path.join(self.__documents_path, filename)

        if not os.path.exists(stored_path):
            
            shutil.copy(file_path, stored_path)

        # use stored path instead of original path
        file_path = stored_path


        loader = PyMuPDFLoader(file_path)

        docs = loader.load()

        pdf = fitz.open(file_path)


        for page_index, doc in enumerate(docs):


            # ---------------- TEXT ----------------


            parent_chunks = self.__parent_splitter.split_text(
                doc.page_content
            )


            for p_id, parent in enumerate(parent_chunks):

                child_chunks = self.__child_splitter.split_text(parent)


                for c_id, child in enumerate(child_chunks):


                    self.__text_collection.add(

                        documents=[child],

                        ids=[

                            f"{filename}_p{page_index}_P{p_id}_C{c_id}"

                        ],

                        metadatas=[{

                            "page": page_index,

                            "document": filename   

                        }]

                    )


            # ---------------- IMAGE ----------------


            page = pdf[page_index]

            pix = page.get_pixmap(dpi=200)

            page_img = Image.open(

                io.BytesIO(pix.tobytes("png"))

            )


            page_area = page_img.width * page_img.height


            results = self.__yolo(page_img, conf=0.5,device=self.__device)


            if not results:

                continue


            result = results[0]


            for det_id, box in enumerate(result.boxes):


                class_id = int(box.cls[0])

                class_name = result.names[class_id]


                if class_name in self.__ignored_layout_classes:

                    continue


                x0, y0, x1, y1 = map(

                    int,
                    box.xyxy[0].tolist()
                )


                crop = page_img.crop(

                    (x0, y0, x1, y1)
                )


                img_path = os.path.join(

                    self.__blob_storage_path,

                    f"{filename}_p{page_index}_fig{det_id}.png"

                )


                crop.save(img_path)

                image_id = str(uuid.uuid4())


                # IMAGE COLLECTION


                self.__image_collection.add(

                    ids=[image_id],

                    uris=[os.path.abspath(img_path)],

                    metadatas=[{

                        "source": img_path,                       

                        "page": page_index,

                        "document": filename  

                    }]

                )


        print(f"PDF indexed correctly: {filename}")
  

    # =====================================================
    # QUERY WITH DOCUMENT FILTER
    # =====================================================


    def query(

        self,

        prompt,

        k_text=6,

        k_image=4,

        document=None

    ):


        where_filter = None


        if document and document != "all":

            where_filter = {

                "document": document

            }


        text_res = self.__text_collection.query(

            query_texts=[prompt],

            n_results=k_text,

            where=where_filter

        )


        img_res = self.__image_collection.query(

            query_texts=[prompt],

            n_results=k_image,

            where=where_filter,

            include=["uris", "metadatas"]

        )


        encoded_images = []

        paths = []


        for uri, meta in zip(

            img_res.get("uris", [[]])[0],

            img_res.get("metadatas", [[]])[0]

        ):


            encoded_images.append(

                encode_image_from_path(uri)
            )


            paths.append(uri)



        return {

            "text":

            text_res.get(

                "documents",

                [[]]

            )[0],


            "images":

            encoded_images,


            "paths":

            paths

        }



    # =====================================================
    # LIST DOCUMENTS
    # =====================================================


    def list_documents(self):

        if not os.path.exists(self.__documents_path):
            return []

        docs = [

            f for f in os.listdir(self.__documents_path)

            if f.lower().endswith(".pdf")

        ]

        return sorted(docs)