from RAG2 import RAGEngine
import chromadb
import os  # for checking if database folder is empty

def main():
    # 1. create a persistent ChromaDB client
    db_path = "./chromadb_storage"
    client = chromadb.PersistentClient(path=db_path)
    rag = RAGEngine(client)

    # -------------------------
    # run first time only: ingest files
    # -------------------------
   
    # print("First run: adding PDFs/images to ChromaDB...")

    # # 3. Add different file types:
    # rag.add_pdf("pdfs\\test2.pdf")
    # rag.add_pdf("pdfs\\test1.pdf")
    # rag.add_pdf("pdfs\\test3.pdf")
    # rag.add_pdf("pdfs\\test4.pdf")
    # rag.add_pdf("pdfs\\test5.pdf")

    # # Images (optional)
    # rag.add_image("Images\\bank.png")
    # rag.add_image("Images\\bear.png")
    # rag.add_image("Images\\cat.png")
    # rag.add_image("Images\\cath.png")
    # rag.add_image("Images\\dog.png")
    # rag.add_image("Images\\mall.png")
    # rag.add_image("Images\\market.png")
    # rag.add_image("Images\\parrot.png")
    # rag.add_image("Images\\rat.png")
    # rag.add_image("Images\\snake.png")
    # rag.add_image("Images\\stadium.png")

    # -------------------------
    # Query
    # -------------------------
    # answer:dict = rag.query("what's happening in stage document preprocessing on mdocagent framework?", 1, 1)
    # answer:dict = rag.query("what's stages of pregenie framework?", 1, 1)
    answer: dict = rag.query("what's architecture of pregenie framework?", 3, 2)
    # answer:dict = rag.query("mouse", 1, 1)

    text: list = answer["text"]
    images: list = answer["images"]

    # -------------------------
    # Print retrieved text
    # -------------------------
    print("Retrieved Text Chunks:")
    for t in text:
        print(t)

    # -------------------------
    # Print retrieved images
    # -------------------------
    print("\nRetrieved Image Paths:")
    for img in images:
        print(img)

    print("\nFull answer dictionary:")
    print(answer)
    print("=====================================\n")


if __name__ == "__main__":
    main()
