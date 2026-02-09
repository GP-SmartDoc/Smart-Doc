import json
import os
import chromadb
# 1. Changed import from ChatGroq to ChatOpenAI
#from langchain_openai import ChatOpenAI 
from langchain_groq import ChatGroq

from RAG import RAGEngine
from SlideGeneration import SlideGenerationModule
import weave # Add this if you want to use weave.init()





def main():
    # Initialize Weave (Newer/Better than just setting env vars)
    #weave.init("gamer7dragon817-ain-shams-university/slide-generator-demo") 

    print("--- Initializing Slide Generation System")
    
    # FIX: Point ChatOpenAI to the W&B Inference Endpoint
    """llm = ChatOpenAI(
        model="meta-llama/Llama-4-Scout-17B-16E-Instruct",
        temperature=0,
        # IMPORTANT: Use the W&B Inference URL
        base_url="https://api.inference.wandb.ai/v1", 
        # Pass your W&B API Key here (or ensure OPENAI_API_KEY is actually your W&B key)
        api_key="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" 
    )"""

    # 4. Initialize Database & RAG
    client = chromadb.PersistentClient(path="./chroma_db")   
    rag = RAGEngine(client)

    # 5. Initialize Slide Generator with OpenAI
    # NOTE: Ensure your SlideGenerationModule class accepts an 'llm' argument!
    slide_gen = SlideGenerationModule(retriever=rag)

    # 6. Add Data
    pdf_path = os.path.join("pdfs", "test1.pdf")
    if os.path.exists(pdf_path):
        print(f"Indexing {pdf_path}...")
        rag.add_pdf(pdf_path)
    else:
        print(f"Warning: {pdf_path} not found. Proceeding with existing DB data.")

    # 7. Define Topic and Run
    topic = "Explain the framework"
    
    # CRITICAL: Lower k_text and k_image to stay under 30k token limit!
    print(f"--- Querying RAG (Reducing context to avoid 413 error)")
    rag_result = rag.query(topic, k_text=2, k_image=1) 
    
    try:
        print("--- [WORKFLOW] Running Multi-Agent Generation...")
        final_output = slide_gen.generate_slides(topic)
        
        # This now uses the new cleaned logic above
        slide_gen.save_as_pptx(final_output, "Presentation.pptx")
        
    except Exception as e:
        print(f"Error during generation: {e}")

if __name__ == "__main__":
    main()