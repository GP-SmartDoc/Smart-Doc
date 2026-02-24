import chromadb
from chromadb.utils import embedding_functions
from chromadb.utils.data_loaders import ImageLoader

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from PIL import Image
import fitz
import io
import os
import uuid
import torch  

from langdetect import detect, DetectorFactory

from ultralytics import YOLO
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download

from transformers import BlipProcessor, BlipForConditionalGeneration

from src.utils.image import encode_image_from_path

class RAGEngine:
    """
    Args:
        chroma_client: Injected client (PersistentClient)
        blob_storage_path: Where to save images extracted from PDFs
    """

    def __init__(self, chroma_client: chromadb.ClientAPI):
        self.__client = chroma_client
        self.__blob_storage_path = "./blob_storage"
        os.makedirs("./blob_storage", exist_ok=True)
        # ---------------- EMBEDDERS ----------------
        self.__english_embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="./models/all-MiniLM-L6-v2"
        )

        self.__arabic_embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="Omartificial-Intelligence-Space/GATE-AraBert-v1"
        )

        self.__image_embedder = embedding_functions.OpenCLIPEmbeddingFunction(
            model_name="ViT-B-32",
            device="cpu"
        )

        self.__image_loader = ImageLoader()

        # ---------------- COLLECTIONS ----------------
        self.__ar_collection = self.__client.get_or_create_collection(
            name="arabic_text", embedding_function=self.__arabic_embedder
        )
        self.__en_collection = self.__client.get_or_create_collection(
            name="english_text", embedding_function=self.__english_embedder
        )

        self.__image_collection = self.__client.get_or_create_collection(
            name="image_collection",
            embedding_function=self.__image_embedder,
            data_loader=self.__image_loader
        )

        # ---------------- TEXT SPLITTING ----------------
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

        # ---------------- IMAGE CAPTIONING ----------------
        

        self.__caption_processor = BlipProcessor.from_pretrained(
            "./models/blip-image-captioning-base"
        )
        self.__caption_model = BlipForConditionalGeneration.from_pretrained(
            "./models/blip-image-captioning-base"
        )
        self.__caption_model.eval()


        # ---------------- IGNORED CLASSES ----------------
        self.__ignored_layout_classes = {
            "Text", "Title", "Section-header", "Page-header",
            "Page-footer", "List-item"
        }

        print("YOLO layout model classes:", self.__yolo.model.names)

    def _get_collection(self, text):
        """Helper to route text based on language"""
        try:
            return self.__ar_collection if detect(text) == 'ar' else self.__en_collection
        except:
            return self.__en_collection # Default to English if detection fails
        
    def add_txt(self, file_path):
        abs_path = os.path.abspath(file_path)
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        
        chunks = self.__child_splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            # Route each chunk to the correct collection
            target_col = self._get_collection(chunk)
            target_col.add(
                documents=[chunk],
                ids=[f"{os.path.basename(file_path)}_chunk_{i}"]
            )
        
    def _caption_image(self, pil_image):
        inputs = self.__caption_processor(
            images=pil_image,
            return_tensors="pt"
        )

        with torch.no_grad():
            out = self.__caption_model.generate(
                **inputs,
                max_new_tokens=50
            )

        caption = self.__caption_processor.decode(
            out[0],
            skip_special_tokens=True
        )

        return caption

    def add_pdf(self, file_path):
        filename = os.path.basename(file_path)
        loader = PyMuPDFLoader(file_path)
        docs = loader.load()
        pdf = fitz.open(file_path)

        for page_index, doc in enumerate(docs):
            # ---------------- TEXT ----------------
            parent_chunks = self.__parent_splitter.split_text(doc.page_content)

            for p_id, parent in enumerate(parent_chunks):
                child_chunks = self.__child_splitter.split_text(parent)

                for c_id, child in enumerate(child_chunks):
                    target_col = self._get_collection(child)
                    target_col.add(
                        documents=[child],
                        ids=[f"{filename}_p{page_index}_P{p_id}_C{c_id}"],
                        metadatas=[{"page": page_index}]
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

                if area_ratio > 0.6:
                    continue
                if w < 200 or h < 200:
                    continue

                aspect_ratio = w / h
                if aspect_ratio > 5 or aspect_ratio < 0.2:
                    continue

                crop = page_img.crop((x0, y0, x1, y1))

                img_path = os.path.join(
                    self.__blob_storage_path,
                    f"{filename}_p{page_index}_fig{det_id}.png"
                )
                crop.save(img_path)

                # -------- IMAGE CAPTIONING --------
                caption = self._caption_image(crop)
                image_id = str(uuid.uuid4())

                # -------- STORE IMAGE --------
                self.__image_collection.add(
                    ids=[image_id],
                    uris=[os.path.abspath(img_path)],
                    metadatas=[{
                        "source": img_path,
                        "page": page_index,
                        "caption": caption
                    }]
                )

                # -------- STORE CAPTION AS TEXT --------
                target_col = self._get_collection(caption)
                target_col.add(
                    documents=[caption],
                    ids=[f"{filename}_p{page_index}_fig{det_id}_caption"],
                    metadatas=[{
                        "type": "image_caption",
                        "image_id": image_id,
                        "page": page_index,
                        "source": img_path
                    }]
                )

        print(f"PDF indexed correctly: {filename}")

    def add_image(self, file_path):
        abs_path = os.path.abspath(file_path)
        image = Image.open(abs_path)
        caption = self._caption_image(image)
        image_id = str(uuid.uuid4())

        self.__image_collection.add(
            ids=[image_id],
            uris=[abs_path],
            metadatas=[{
                "source": abs_path,
                "caption": caption
            }]
        )

        target_col = self._get_collection(caption)
        target_col.add(
            documents=[caption],
            ids=[f"{image_id}_caption"],
            metadatas={
                "type": "image_caption",
                "image_id": image_id,
                "source": abs_path
            }
        )

    def add_file(path:str):
        """
        TO BE IMPLEMENTED: add any file type, then a specific add fucntion
            should be called based on file extension. If a file type is unavailible, 
            such case should be handeled.
        """
        print()
        pass

    def query(self, prompt, k_text=5, k_image=3):
        """Returns top-k most relevent text chunks, and top-k most relevent images to a user's prompt

        Args:
            prompt (str): The user's prompt
            k_text (int): Number of text chunks to retrieve
            k_image (int): Number of images to retrieve

        Returns:
            a dictionary in the form {"text": list[str], "images":list[str]} conntaining the 
            retrieved text chunks and retrieved image paths
        """
        # Detect prompt language to know which collection to search
        target_col = self._get_collection(prompt)

        text_res = target_col.query(
        query_texts=[prompt],
        n_results=k_text
    )

        img_res = self.__image_collection.query(
            query_texts=[prompt],
            n_results=k_image,
            include=["uris", "metadatas"]
        )

        encoded_images = []
        image_captions = []
        paths = []   # ✅ NEW

        for uri, meta in zip(
            img_res.get("uris", [[]])[0],
            img_res.get("metadatas", [[]])[0]
        ):
            encoded_images.append(encode_image_from_path(uri))
            image_captions.append(meta.get("caption", ""))
            paths.append(uri)   

        return {
            "text": text_res.get("documents", [[]])[0],
            "images": encoded_images,
            "image_captions": image_captions,
            "paths": paths     
        }