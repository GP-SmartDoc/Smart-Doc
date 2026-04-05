import os
import chromadb

import re
import textwrap
from src.config.model import text_model as model
from src.config.model import image_model as v_model
from src.vector_store.RAG import RAGEngine
from src.graphs.summary_graph import SummarizationModule
from src.graphs.qa_graph import QuestionAnsweringModule
from src.graphs.slide_generation_graph import generate_slides
from src.graphs.visualization_graph import generate_visualization
from src.states.visualization_state import DiagramType

from langchain.messages import SystemMessage, HumanMessage
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from src.vector_store.RAG import RAGEngine
from src.graphs.summary_graph import SummarizationModule
from src.graphs.qa_graph import QuestionAnsweringModule
from src.graphs.slide_generation_graph import generate_slides
from src.utils.pptx import save_as_pptx

# NEW
from src.memory.chat_memory import ChatMemory

# ----------------------------
# Request Model
# ----------------------------
def detect_intent_model(user_input: str) -> str:
    """
    Ask the LLM to classify the user input intent.
    Returns one of: 'summary', 'visualization', 'qa', or 'slide_generation'.
    """
    prompt = f"""
Determine the intent of the following user input.
Return ONLY one word: 'summary', 'visualization', 'qa', or 'slide_generation'.

User Input:
\"\"\"{user_input}\"\"\"
"""
    resp = model.invoke([
        SystemMessage(content="You are an assistant that classifies user intent."),
        HumanMessage(content=prompt)
    ])
    
    # Normalize response
    intent = resp.content.strip().lower()
    if intent not in ["summary", "visualization", "qa","slide_generation"]:
        return "qa"  # fallback default
    return intent

def visualization_module(prompt: str):
    """
    Calls visualization graph and returns diagram text.
    Default diagram type can be changed depending on user prompt.
    """
    
    # simple auto-detection of diagram type from text
    text = prompt.lower()

    if "flowchart" in text:
        diagram_type = DiagramType.FLOWCHART
        
    elif "state diagram" in text or "state machine" in text:
        diagram_type = DiagramType.STATE
        
    elif "class diagram" in text:
        diagram_type = DiagramType.CLASS
        
    elif "er" in text or "entity relationship" in text:
        diagram_type = DiagramType.ER
        
    elif "pie" in text or "chart" in text:
        diagram_type = DiagramType.PIE
        
    elif "mindmap" in text or "mind map" in text:
        diagram_type = DiagramType.MINDMAP
        
    else:
        # default fallback
        diagram_type = DiagramType.MINDMAP

    return generate_visualization(
        type=diagram_type,
        description=prompt
    )
# ----------------------------
# GUI
# ----------------------------
app = FastAPI()
templates = Jinja2Templates(directory="templates")

class ChatRequest(BaseModel):
    message: str
    document: str
    mode: str
    summary_mode: str = "overview"

# ----------------------------
# App and Templates
# ----------------------------
app = FastAPI()
templates = Jinja2Templates(directory="templates")

UPLOAD_FOLDER = "documents"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ----------------------------
# Initialize System
# ----------------------------
print("--- Initializing Smart Doc System ---")
try:
    # 1. ChromaDB Client
    client = chromadb.PersistentClient(path="./chroma_db")
    # 2. RAG Engine
    rag = RAGEngine(client)
    # 3. Initialize QA and Summary modules
    qa_module = QuestionAnsweringModule(retriever=rag)
    summary_module = SummarizationModule(retriever=rag)
    slide_generation_module = lambda prompt, document: generate_slides(rag, prompt, document=document)
    memory = ChatMemory()  # NEW: Initialize chat memory with a max length of 10 interactions
    #visualization_module = lambda state: "Visualization module is under development. Please check back later."
    print("System Ready.\n")
except Exception as e:
    print(f"Initialization Error: {e}")

# ----------------------------
# Routes
# ----------------------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )

@app.get("/documents")
def list_documents():
    return {
    "documents":rag.list_documents()
    }

@app.post("/send")
def receive_message(data: ChatRequest):

    user_msg = data.message
    document = data.document
    mode = data.mode

    # ----------------------------
    # Add user message to memory
    # ----------------------------
    memory.add("user", user_msg)
    print("\n--- CHAT MEMORY AFTER USER MESSAGE ---")
    for i, m in enumerate(memory.history):
        print(f"{i+1}. {m['role']}: {m['content'][:80]}")
    print("--------------------------------------\n")
    # Build context-aware question
    enhanced_question = memory.build_context(user_msg)
    print("\n--- ENHANCED QUESTION SENT TO MODEL ---")
    print(enhanced_question)
    print("---------------------------------------\n")
    # ----------------------------
    # QA Mode
    # ----------------------------
    if mode == "qa":

        result = qa_module.invoke(
            question=enhanced_question,
            document=document
        )

        clean_answer = result["Answer"][12:-2]

        reply = format_qa_output(clean_answer)

    # ----------------------------
    # Summary Mode
    # ----------------------------
    elif mode == "summary":

        summary_mode = data.summary_mode

        print(f"Invoking summary with mode: {summary_mode}")

        result = summary_module.invoke(
            question=enhanced_question,
            document=document,
            summary_mode=summary_mode
        )

        clean_answer = result["Answer"]

        reply = format_summarize_output(clean_answer)

    # ----------------------------
    # Slide Generation
    # ----------------------------
    elif mode == "slide_generation":

        reply = slide_generation_module(
            enhanced_question,
            document=document
        )

        save_as_pptx(
            reply,
            "layouts.pptx",
            "generated_slides.pptx"
        )

        reply = "Slide generation completed and saved as 'generated_slides.pptx'."

    # ----------------------------
    # Visualization
    # ----------------------------
    elif mode == "visualization":

        reply = visualization_module(user_msg)

    else:

        reply = f"You said: {user_msg}"

    # ----------------------------
    # Save assistant reply
    # ----------------------------
    memory.add("assistant", reply)
    print("\n--- CHAT MEMORY AFTER ASSISTANT RESPONSE ---")
    for i, m in enumerate(memory.history):
        print(f"{i+1}. {m['role']}: {m['content'][:80]}")
    print("--------------------------------------------\n")
    return {"reply": reply}

def format_qa_output(raw_text: str) -> str:
    """
    Format raw QA output to professional style:
    - Numbered points (1., (1)) start on new lines
    - Bullets (*, -, •) start on new lines
    - Each component/action appears on its own line
    - Sentences are flowing and paragraphs are preserved
    """
    # answer = clean_answer.strip()

    # # Replace escaped newlines with actual newlines
    # answer = answer.replace("\\n", "\n")

    # # Ensure numbered lists start on a new line
    # answer = re.sub(r'(?<!\d)(\d+)\.', r'\n\1.', answer)

    # # Replace "*" bullets with "•" and ensure single newline before each
    # answer = re.sub(r'^\s*\*\s*', r'• ', answer, flags=re.MULTILINE)

    # # Remove excessive blank lines (more than 2)
    # answer = re.sub(r'\n{3,}', r'\n\n', answer)

    # reply = answer.strip()

    text = raw_text.strip()

    # fix escaped newlines
    text = text.replace("\\n", "\n")

    # --------------------
    # fix broken bold markdown
    # --------------------

    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)

    text = re.sub(r'\*(.*?)\*', r'\1', text)

    # remove single leftover *
    text = re.sub(r'\*+', '', text)

    # numbered list: 1.
    text = re.sub(r'\s*(\d+)\.\s+', r'\n\1. ', text)

    # numbered list: (1)
    text = re.sub(r'\s*\((\d+)\)\s*', r'\n(\1) ', text)


    # --------------------
    # fix dash bullets 
    # --------------------

    text = re.sub(r'\s-\s+(?=[A-Z])', r'\n• ', text)

    # --------------------
    # fix bullet symbols
    # --------------------

    text = re.sub(r'\s*•\s+', r'\n• ', text)

    # --------------------
    # clean spacing
    # --------------------

    text = re.sub(r'\n{2,}', '\n\n', text)

    text = re.sub(r'[ \t]+', ' ', text)

    # --------------------
    # final cleanup
    # --------------------

    lines = [line.strip() for line in text.split('\n')]

    text = "\n".join(line for line in lines if line)

    return text.strip()





def format_summarize_output(raw_text: str) -> str:
    answer = raw_text.strip()

    # 1. Replace escaped newlines with spaces
    answer = answer.replace("\\n", " ")

    # 2. Fix line breaks in the middle of numbers or model names (e.g., Qwen 2.\n5 → Qwen 2.5)
    answer = re.sub(r'(\b\d)\.\s*\n\s*(\d)', r'\1.\2', answer)

    # 3. Remove multiple spaces
    answer = re.sub(r'\s+', ' ', answer)

    # 4. Optionally, split into logical paragraphs (2 sentences per paragraph)
    sentences = re.split(r'(?<=[.!?]) +', answer)
    paragraphs = [" ".join(sentences[i:i+2]) for i in range(0, len(sentences), 2)]
    formatted_answer = "\n\n".join(paragraphs)

    # 5. Strip leading/trailing spaces
    reply = formatted_answer.strip()

    return reply




@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_path = f"{UPLOAD_FOLDER}/{file.filename}"

    try:
        with open(file_path, "wb") as f:
            f.write(await file.read())

        rag.add_pdf(file_path)

        print("PDF successfully indexed.")

        return {"status": f"✅ {file.filename} uploaded"}

    except Exception as e:
        return {"status": f"❌ Upload failed: {str(e)}"}
