import chromadb
from chromadb.utils import embedding_functions
from chromadb.utils.data_loaders import ImageLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PIL import Image
import os
import io
import uuid
import fitz  # PyMuPDF

class RAGEngine:
    """
    Args:
        chroma_client: Injected client (PersistentClient)
        blob_storage_path: Where to save images extracted from PDFs
    """
    def __init__(self, chroma_client: chromadb.ClientAPI, blob_storage_path="./blob_storage"):
        # self.__reset()
        self.__client = chroma_client 
        self.__blob_storage_path = blob_storage_path
        
        # Ensure blob storage exists
        os.makedirs(self.__blob_storage_path, exist_ok=True)
        
        # Image Loader
        self.__image_loader = ImageLoader()
        
        # Embedders
        self.__text_embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        # self.__image_embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
        #     model_name="sentence-transformers/clip-ViT-B-32"
        # )
        self.__image_embedder = embedding_functions.OpenCLIPEmbeddingFunction()

        # Collections
        self.__text_collection = self.__client.get_or_create_collection(
            name="text_collection",
            embedding_function=self.__text_embedder
        )
        self.__image_collection = self.__client.get_or_create_collection(
            name="image_collection", 
            embedding_function=self.__image_embedder,
            data_loader=self.__image_loader
        )
        
        # Text Splitter
        self.__splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    def add_txt(self, file_path):
        abs_path = os.path.abspath(file_path)
        with open(abs_path, "r", encoding="utf-8") as f:
            text = f.read()
        
        # Proper Chunking
        chunks = self.__splitter.split_text(text)
        
        if chunks:
            ids = [f"{os.path.basename(file_path)}_chunk_{i}" for i in range(len(chunks))]
            self.__text_collection.add(documents=chunks, ids=ids)
            print(f"Added {len(chunks)} text chunks from {file_path}")
        
    def add_pdf(self, file_path):
        loader = PyMuPDFLoader(file_path)
        docs = loader.load()
        filename = os.path.basename(file_path)

        pdf = fitz.open(file_path)

        for i, doc in enumerate(docs):
            # TEXT EXTRACTION
            text = doc.page_content
            chunks = self.__splitter.split_text(text)
            if chunks:
                ids = [f"{filename}_p{i}_c{j}" for j in range(len(chunks))]
                self.__text_collection.add(documents=chunks, ids=ids)

            # PAGE RENDER
            page = pdf[i]
            pix = page.get_pixmap(matrix=fitz.Matrix(2,2))
            page_image = Image.open(io.BytesIO(pix.tobytes("png")))

            # AUTO DIAGRAM DETECTION
            diagrams = self.extract_diagrams_from_page(page, page_image)

            for idx, img in diagrams:
                save_path = os.path.join(
                    self.__blob_storage_path,
                    f"{filename}_diagram_{i}_{idx}.png"
                )
                img.save(save_path)
                self.add_image(save_path)

        print("PDF processed:", file_path)

    
    def add_image(self, file_path):
        """Adds an image by URI. The Embedder loads the file from disk."""
        abs_path = os.path.abspath(file_path)
        if not os.path.exists(abs_path):
            print(f"Image not found: {abs_path}")
            return

        # Chroma's OpenCLIP loader will read the file at 'uri'
        self.__image_collection.add(
            ids=[str(uuid.uuid4())],
            uris=[abs_path],
            metadatas=[{"source": abs_path}]
        )
        print(f"Indexed Image: {file_path}")

    def add_file(path:str):
        """
        TO BE IMPLEMENTED: add any file type, then a specific add fucntion
            should be called based on file extension. If a file type is unavailible, 
            such case should be handeled.
        """
        print()
        pass
    
    def query(self, prompt:str, k_text:int, k_image:int) -> dict:
        """Returns top-k most relevent text chunks, and top-k most relevent images to a user's prompt

        Args:
            prompt (str): The user's prompt
            k_text (int): Number of text chunks to retrieve
            k_image (int): Number of images to retrieve

        Returns:
            a dictionary in the form {"text": list[str], "images":list[str]} conntaining the 
            retrieved text chunks and retrieved image paths
        """
        # Defaults
        retrieved_text = []
        retrieved_image_paths = []

        # --- FIX: Only query if k > 0 ---
        if k_text > 0:
            text_results = self.__text_collection.query(
                query_texts=[prompt],
                n_results=k_text
            )
            # text_results["documents"] is a list of lists (one list per query)
            if text_results.get("documents") and len(text_results["documents"]) > 0:
                retrieved_text = text_results["documents"][0]
        
        if k_image > 0:
            image_results = self.__image_collection.query(
                query_texts=[prompt],
                n_results=k_image,
                include=["uris", "distances"]
            )
            if image_results.get("uris") and len(image_results["uris"]) > 0:
                retrieved_image_paths = image_results["uris"][0]
        
        return {
            "text": retrieved_text,
            "images": retrieved_image_paths 
        }
    
    def __reset(self):
        # ---------------------------------------------------------
        # CRITICAL FIX: RESET COLLECTION TO APPLY DATA LOADER
        # ---------------------------------------------------------
        # Only run this ONCE or when you change configuration/schema. 
        # If you keep this, it wipes data every restart. 
        # For dev, you can check if it exists or catch error.
        try:
            self.__client.delete_collection("image_collection")
            print("Deleted old image_collection to apply new data_loader config.")
        except:
            pass
        
        try:
            self.__client.delete_collection("text_collection")
            print("Deleted old image_collection to apply new data_loader config.")
        except:
            pass


    def extract_diagrams_from_page(self, page, page_image):
        drawings = page.get_drawings()
        boxes = []

        # 1) Collect bounding boxes of vector drawings
        for d in drawings:
            if "rect" in d and d["rect"] is not None:
                rect = d["rect"]
                x0, y0, x1, y1 = rect
                w = abs(x1 - x0)
                h = abs(y1 - y0)

                # Skip tiny elements (icons, dots)
                if w < 100 or h < 100:
                    continue

                boxes.append(rect)

        if not boxes:
            return []  # No diagrams found

        # 2) Merge overlapping bounding boxes
        def merge_boxes(boxes):
            merged = True
            while merged:
                merged = False
                new_boxes = []
                while boxes:
                    a = boxes.pop(0)
                    ax0, ay0, ax1, ay1 = a
                    overlap_found = False

                    for b in boxes:
                        bx0, by0, bx1, by1 = b

                        # Check overlap
                        if not (ax1 < bx0 or ax0 > bx1 or ay1 < by0 or ay0 > by1):
                            # Merge
                            new_box = (
                                min(ax0, bx0),
                                min(ay0, by0),
                                max(ax1, bx1),
                                max(ay1, by1),
                            )
                            boxes.remove(b)
                            boxes.append(new_box)
                            overlap_found = True
                            merged = True
                            break

                    if not overlap_found:
                        new_boxes.append(a)

                boxes = new_boxes
            return boxes

        merged_boxes = merge_boxes(boxes)

        # 3) Crop diagrams from page image
        diagram_images = []
        for i, (x0, y0, x1, y1) in enumerate(merged_boxes):
            crop = page_image.crop((x0*2, y0*2, x1*2, y1*2))  # multiply by scale 2
            diagram_images.append((i, crop))

        return diagram_images