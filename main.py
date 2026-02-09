# main.py
import os
import sys
import chromadb
from src.config.model import model
from src.vector_store.RAG import RAGEngine
from src.graphs.summary_graph import SummarizationModule
from src.graphs.qa_graph import QuestionAnsweringModule

# ----------------------------
# Intent Detection
# ----------------------------
def detect_intent(user_input: str) -> str:
    summary_keywords = [
        "summarize", "summary", "overview", "tl;dr", "abstract",
        "give me a summary", "summarise"
    ]
    visualization_keywords = [
        "diagram", "chart", "plot", "visualize", "visualisation", "graph"
    ]
    q = user_input.lower()
    if any(k in q for k in summary_keywords):
        return "summary"
    if any(k in q for k in visualization_keywords):
        return "visualization"
    return "qa"

# ----------------------------
# Main Function
# ----------------------------
def main():
    print("--- Initializing Smart Doc System ---")
    try:
        # 1. ChromaDB Client
        client = chromadb.PersistentClient(path="./chroma_db")

        # 2. RAG Engine
        rag = RAGEngine(client)

        # 3. Initialize QA and Summary modules
        qa_module = QuestionAnsweringModule(retriever=rag, model=model)
        summary_module = SummarizationModule(retriever=rag, model=model)

        print("System Ready.\n")

    except Exception as e:
        print(f"Initialization Error: {e}")
        return

    # ----------------------------
    # Main Loop
    # ----------------------------
    while True:
        print("=" * 50)
        pdf_input = input("Enter path to PDF file (or type 'exit' to quit): ").strip()
        pdf_path = pdf_input.strip('"').strip("'")

        if pdf_path.lower() == 'exit':
            print("Goodbye!")
            break

        if not os.path.exists(pdf_path):
            print(f"Error: File not found at '{pdf_path}'. Please try again.")
            continue

        # Reset collections per document if supported
        try:
            rag.reset_collections()
        except AttributeError:
            pass

        # Add PDF to RAG
        print(f"\nProcessing '{os.path.basename(pdf_path)}'...")
        try:
            rag.add_pdf(pdf_path)
            print("PDF successfully indexed.")
        except Exception as e:
            print(f"Error processing PDF: {e}")
            continue

        # ----------------------------
        # Question Loop
        # ----------------------------
        while True:
            print("-" * 30)
            question = input("\nEnter your question (or 'new' for new PDF, 'exit' to quit): ").strip()

            if question.lower() == 'exit':
                print("Goodbye!")
                sys.exit(0)
            if question.lower() == 'new':
                print("Returning to file selection...")
                break
            if not question:
                continue

            intent = detect_intent(question)
            print(f"\nDetected intent: {intent.upper()}")
            print("Processing...\n")

            try:
                # ----------------------------
                # Intent Routing
                # ----------------------------
                if intent == "summary":
                    final_answer = summary_module.invoke(question)
                    print("\n" + "#" * 16 + " DOCUMENT SUMMARY " + "#" * 16)

                elif intent == "visualization":
                    final_answer = (
                        "Visualization pipeline is not implemented yet.\n"
                        "However, the system successfully detected visualization intent."
                    )
                    print("\n" + "#" * 14 + " VISUALIZATION " + "#" * 14)

                else:  # QA
                    final_answer = qa_module.invoke(question)
                    print("\n" + "#" * 20 + " ANSWER " + "#" * 20)

                print(final_answer)
                print("#" * 48 + "\n")

            except Exception as e:
                print(f"An error occurred while processing the request: {e}")


# ----------------------------
# Run the main function
# ----------------------------
if __name__ == "__main__":
    main()
