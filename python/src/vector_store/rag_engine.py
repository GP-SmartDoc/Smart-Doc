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
from pathlib import Path

'''
Notes:
    1. chromadb.Client() should be changed to PersistentClient() later. Client() is in-memory.
    2. The RAGEngine should not create the Client object, it should get injected to it.
        chromadb is known for concurency bug and having multiple clinets will cause such triuvle
    3. An even better practice than (2) is to run chroma as in individual service and access it 
        through an API. Should put some thought into that
    4. Whys is add_image()'s logic slightly different than addding images in add_pdf() ?
'''

class RAGEngine:
    """
    Args:
        chroma_client: Injected client (PersistentClient)
        blob_storage_path: Where to save images extracted from PDFs
    """
    def __init__(self, chroma_client: chromadb.ClientAPI): #, blob_storage_path="./blob_storage"):
        # self.__reset()
        self.__client = chroma_client 
        # self.__blob_storage_path = blob_storage_path
        self.__blob_storage_path = Path(__file__).parent / "blob_storage" # <-- this should be accessed from .env file and not hard-coded!!!!
        
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
        """Extracts text AND saves images to disk for indexing"""
        loader = PyMuPDFLoader(file_path)
        docs = loader.load()
        
        filename = os.path.basename(file_path)

        for i, doc in enumerate(docs):
            # 1. Process Text
            page_text = doc.page_content
            chunks = self.__splitter.split_text(page_text)
            if chunks:
                ids = [f"{filename}_p{i}_c{j}" for j in range(len(chunks))]
                self.__text_collection.add(documents=chunks, ids=ids)

            # 2. Extract Images (PyMuPDF Logic)
            # Note: PyMuPDFLoader might require manual image extraction logic depending on version,
            # but assuming standard fitz access:
            try:
                # We assume extract_images is handled or we use the 'fitz' library directly here
                # For simplicity in this example, we assume we can get image bytes.
                # If using LangChain's loader, images might not be in metadata by default.
                # Ideally, iterate using `fitz` directly for image extraction:
                import fitz 
                pdf_file = fitz.open(file_path)
                for page_index in range(len(pdf_file)):
                    page = pdf_file[page_index]
                    image_list = page.get_images()
                    
                    for img_index, img in enumerate(image_list):
                        xref = img[0]
                        base_image = pdf_file.extract_image(xref)
                        image_bytes = base_image["image"]
                        
                        # SAVE IMAGE TO BLOB STORAGE
                        image_name = f"{filename}_p{page_index}_i{img_index}.png"
                        save_path = os.path.join(self.__blob_storage_path, image_name)
                        
                        with open(save_path, "wb") as f:
                            f.write(image_bytes)
                            
                        # ADD TO CHROMA using the URI
                        self.add_image(save_path)
            except Exception as e:
                print(f"Error extracting images from PDF: {e}")
        print(f"Processed PDF: {file_path}")
    
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
        text_results = self.__text_collection.query(
            query_texts=[prompt],
            n_results=k_text
        )
        retrieved_text = text_results["documents"][0] if text_results["documents"] else ""
            
        image_results = self.__image_collection.query(
            query_texts=prompt,
            n_results=k_image,
            include=["uris", "distances"]
        )
        retrieved_image_paths = image_results["uris"][0] if image_results["uris"] else []
        
        return {
            "text": retrieved_text,
            "images": retrieved_image_paths 
        }
    
    def __reset(self):
        # ---------------------------------------------------------
        # RESET COLLECTION TO APPLY DATA LOADER
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


        