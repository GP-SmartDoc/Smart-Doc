# main.py
import os
import sys
import chromadb
import re
import textwrap
from src.config.model import model
from src.vector_store.RAG import RAGEngine
from src.graphs.summary_graph import SummarizationModule
from src.graphs.qa_graph import QuestionAnsweringModule
from src.graphs.slide_generation_graph import generate_slides
from langchain.messages import SystemMessage, HumanMessage
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates


from src.utils.pptx import save_as_pptx



# ----------------------------
# Model-driven Intent Detection
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




# ----------------------------
# GUI
# ----------------------------
app = FastAPI()
templates = Jinja2Templates(directory="templates")

UPLOAD_FOLDER = "documents"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ----------------------------
# Main
# ----------------------------
print("--- Initializing Smart Doc System ---")
try:
    # 1. ChromaDB Client
    client = chromadb.PersistentClient(path="./chroma_db")

    # 2. RAG Engine
    rag = RAGEngine(client)

    # 3. Initialize QA and Summary modules
    qa_module = QuestionAnsweringModule(retriever=rag, model=model)
    summary_module = SummarizationModule(retriever=rag, model=model)
    visualization_module = lambda prompt, document: generate_slides(rag, prompt, document=document)
    print("System Ready.\n")
except Exception as e:
    print(f"Initialization Error: {e}")

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# ------------------
# LIST DOCUMENTS
# ------------------

@app.get("/documents")
def list_documents():
    return {
    "documents":rag.list_documents()
    }

@app.post("/send")
def receive_message(data: dict):
    user_msg = data.get("message", "")
    mode = data.get("mode", "qa")  # default to "qa"
    document = data.get("document", "all")  # default to "all"
    ##########################################
    """#########  Call LLM Here Based on Mode ##########"""
    ##########################################

    if mode == "qa":
        result = qa_module.invoke(question=user_msg, document=document)
        clean_answer = result["Answer"][12:-2]
        reply = format_qa_output(clean_answer)
    elif mode == "summarize":
        result = summary_module.invoke(question=user_msg, document=document)
        clean_answer = result['Answer']
        reply = format_summarize_output(clean_answer)
    elif mode == "viz":
        reply = visualization_module(user_msg, document=document)
        save_as_pptx(reply, "generated_slides.pptx")
        reply = "Visualization generated and saved as 'generated_slides.pptx'."
    else:
        reply = f"You said: {user_msg}"
    
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

    answer = raw_text.strip()

    # Replace escaped newlines with space
    answer = answer.replace("\\n", " ")

    # Ensure numbered points "1." start on a new line
    answer = re.sub(r'\s*(\d+)\.\s*', r'\n\1. ', answer)

    # Ensure numbered points "(1)" start on a new line
    answer = re.sub(r'\s*\(\s*(\d+)\s*\)\s*', r'\n(\1) ', answer)

    # Convert bullets at start of line to •, ignore hyphens in words
    answer = re.sub(r'^\s*[\*\-\•]\s+', r'• ', answer, flags=re.MULTILINE)

    # Split into sentences while keeping numbering/bullets together
    sentences = re.split(r'(?<!\d)\.(\s+)', answer)

    # Rejoin sentences on separate lines
    formatted_sentences = []
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        # Keep numbered points and bullets intact
        formatted_sentences.append(s if re.match(r'^(\d+\.|\(\d+\)|• )', s) else s + '.')

    formatted_answer = "\n".join(formatted_sentences)

    # Remove multiple blank lines
    formatted_answer = re.sub(r'\n{2,}', r'\n\n', formatted_answer)

    return formatted_answer.strip()




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
    except Exception as e:
        print(f"Error processing PDF: {e}")
        return

    ##########################################
    """#######  Process File Here  ########"""
    ##########################################

    return {"status": f"✅ {file.filename} uploaded"}