from RAG import RAGEngine
import chromadb

def main():
    # 1. create a chromadbclient
    client = chromadb.PersistentClient()   
    rag = RAGEngine(client)

    # run first time only
    #3.Add different file types:
    rag.add_pdf("pdfs\\test2.pdf")
    rag.add_pdf("pdfs\\test1.pdf")
    rag.add_pdf("pdfs\\test3.pdf")
    rag.add_pdf("pdfs\\test4.pdf")
    rag.add_pdf("pdfs\\test5.pdf")

    rag.add_image("Images\\bank.png")
    rag.add_image("Images\\bear.png")
    rag.add_image("Images\\cat.png")
    rag.add_image("Images\\cath.png")
    rag.add_image("Images\\dog.png")
    rag.add_image("Images\\mall.png")
    rag.add_image("Images\\market.png")
    rag.add_image("Images\\parrot.png")
    rag.add_image("Images\\rat.png")
    rag.add_image("Images\\snake.png")
    rag.add_image("Images\\stadium.png")

    #answer:dict = rag.query("what's happening in stage document preprocessing on mdocagent framework?", 1, 1)
    answer:dict = rag.query("what's stages of pregenie framework?", 1, 1)
    #answer:dict = rag.query("mouse", 1, 1)

    text:list = answer["text"] 
    images:list = answer["images"] 
    
    print("Retrieved Text Chunks:")
    for t in text:
        print(t)
    print("\nRetrieved Image Paths:")
    for img in images:
        print(img)
    print("\n")
    print(answer)

if __name__ == "__main__":
    main()