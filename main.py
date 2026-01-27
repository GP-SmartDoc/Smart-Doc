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
    topic = "Explain the stages of the framework"
    print(f"\n--- Generating Slides for: '{topic}' ---\n")
    
    try:
        final_code = slide_gen.generate_slides(topic)
        
        # Save Output
        output_file = "slides.md"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(final_code)
            
        print(f"\nSuccess! Slidev code saved to '{output_file}'")
        print("\n--- Preview of Generated Code ---\n")
        print(final_code[:500] + "...\n(truncated)")
        #print("\nCheck W&B dashboard for execution traces.")
        
    except Exception as e:
        print(f"Error during generation: {e}")

if __name__ == "__main__":
    main()