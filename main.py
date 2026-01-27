from RAG import RAGEngine
from QuestionAnswering import QuestionAnsweringModule
#from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
import chromadb
import os
import sys

def main():
    # 1. Initialize System Components
    print("--- Initializing Smart Doc System ---")
    try:
        # Initialize ChromaDB Client
        client = chromadb.PersistentClient(path="./chroma_db")   
        
        # Initialize RAG Engine
        rag = RAGEngine(client)

        # Initialize QA Module with ChatOllama
        llm_model = ChatOllama(model="qwen3-vl:4b", temperature=0)
        qa_module = QuestionAnsweringModule(retriever=rag, model=llm_model)
        
        print("System Ready.\n")

    except Exception as e:
        print(f"Initialization Error: {e}")
        return

    # 2. Main Execution Loop
    while True:
        print("="*50)
        pdf_input = input("Enter path to PDF file (or type 'exit' to quit): ").strip()
        
        # Clean up path (remove quotes often added by drag-and-drop)
        pdf_path = pdf_input.strip('"').strip("'")

        if pdf_path.lower() == 'exit':
            print("Goodbye!")
            break

        if not os.path.exists(pdf_path):
            print(f"Error: File not found at '{pdf_path}'. Please try again.")
            continue
        
        # 3. Add PDF to RAG
        print(f"\nProcessing '{os.path.basename(pdf_path)}'...")
        try:
            rag.add_pdf(pdf_path)
            print("PDF successfully indexed.")
        except Exception as e:
            print(f"Error processing PDF: {e}")
            continue

        # 4. Question Loop for the current PDF
        while True:
            print("-" * 30)
            question = input("\nEnter your question (or type 'new' for new PDF, 'exit' to quit): ").strip()

            if question.lower() == 'exit':
                print("Goodbye!")
                sys.exit(0) # Exit the entire program
            
            if question.lower() == 'new':
                print("Returning to file selection...")
                break # Break inner loop to go back to PDF selection
            
            if not question:
                continue

            print(f"\nAnalyzing question...")
            try:
                final_answer = qa_module.invoke_workflow(question)
                print("\n" + "#"*20 + " ANSWER " + "#"*20)
                print(final_answer)
                print("#"*48 + "\n")
            except Exception as e:
                print(f"An error occurred while answering: {e}")

if __name__ == "__main__":
    main()