import os
import shutil
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain.messages import AnyMessage, HumanMessage, AIMessage

from src.nodes.slide_generation.text_summarizer_agent import Text_Summarizer
from src.nodes.slide_generation.image_captioner_agent import Image_Captioner
from src.nodes.slide_generation.code_generator_agent import Code_Generator
from src.nodes.slide_generation.page_reviewer_agent import Page_Reviewer
from src.nodes.slide_generation.code_reviewer_agent import Code_Reviewer
from src.nodes.slide_generation.code_generator_reviewed_agent import Code_Generator_Reviewed
from src.states.SGState import SlideGenerationGraphState


builder = StateGraph(SlideGenerationGraphState)

builder.add_node("Text_Summarizer", Text_Summarizer)
builder.add_node("Image_Captioner", Image_Captioner)
builder.add_node("Code_Generator", Code_Generator)
builder.add_node("Code_Reviewer", Code_Reviewer)
builder.add_node("Page_Reviewer", Page_Reviewer)
builder.add_node("Code_Generator_Reviewed", Code_Generator_Reviewed)

builder.add_edge(START, "Text_Summarizer")
builder.add_edge("Text_Summarizer", "Image_Captioner")
builder.add_edge("Image_Captioner", "Code_Generator")
builder.add_edge("Code_Generator", "Code_Reviewer")
builder.add_edge("Code_Reviewer", "Page_Reviewer")
builder.add_edge("Page_Reviewer", "Code_Generator_Reviewed")
builder.add_edge("Code_Generator_Reviewed", END)

sg_module = builder.compile()

def generate_slides(rag,prompt:str, document: str = "all")->str:
    """
    Main entry point: Queries RAG, copies images to local folder, then runs generation.
    """
    retrieved_data = rag.query(prompt, k_text=6, k_image=5, document=document)
    
    text_content = retrieved_data.get("text")

    retrieved_text = "\n".join(text_content)
    retrieved_images = retrieved_data.get("paths")
    
    # --- CRITICAL FIX: Copy images to local 'images/' folder ---
    # This fixes the absolute path/backslash error in Slidev
    local_image_paths = []
    
    if retrieved_images:
        os.makedirs("images", exist_ok=True) # Create folder
        #print(f"Copying {len(retrieved_images)} images to local 'images/' folder...")
        
        for src_path in retrieved_images:
            if os.path.exists(src_path):
                filename = os.path.basename(src_path)
                # Sanitize filename (remove spaces, etc. if needed)
                filename = filename.replace(" ", "_")
                
                dst_path = os.path.join("images", filename)
                shutil.copy(src_path, dst_path)
                
                # Store as FORWARD SLASH path for Markdown
                # e.g., "images/my_diagram.png"
                local_image_paths.append(f"images/{filename}")
            else:
                print(f"Warning: Source image not found: {src_path}")
    
    if not text_content:
        text_content = "No specific text found in documents."
        print("Warning: No text retrieved from RAG.")

    initial_state = {
        "messages": [HumanMessage(content=prompt)],
        "llm_calls": 0,
        "retrieved_text": retrieved_text,
        "retrieved_images": local_image_paths, # Pass the CLEAN local paths
        "document": document,
        "Text_Summarizer_output": "",
        "Image_Captioner_output": "",
        "Code_Generator_output": "",
        "Code_Reviewer_output": "",
        "Page_Reviewer_output": "",
        "Code_Generator_output_Reviewed": ""
    }
    
    print("Starting Multi-Agent Workflow...")
    final_state = sg_module.invoke(initial_state)
    return final_state.get("Code_Generator_output_Reviewed", "")