# main.py
import os
import chromadb
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from src.config.model import model
from src.vector_store.RAG import RAGEngine
from src.graphs.summary_graph import SummarizationModule
from src.graphs.qa_graph import QuestionAnsweringModule
from src.graphs.slide_generation_graph import generate_slides
from src.utils.pptx import save_as_pptx

# ----------------------------
# Request Model
# ----------------------------
class ChatRequest(BaseModel):
    message: str
    document: str
    mode: str
    summary_mode: str = "overview"  # <-- added for summary button

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
client = chromadb.PersistentClient(path="./chroma_db")
rag = RAGEngine(client)
qa_module = QuestionAnsweringModule(retriever=rag, model=model)
summary_module = SummarizationModule(retriever=rag, model=model)
slide_generation_module = lambda prompt, document: generate_slides(rag, prompt, document=document)
visualization_module = lambda state: "Visualization module is under development. Please check back later."
print("System Ready.\n")

# ----------------------------
# Routes
# ----------------------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/documents")
def list_documents():
    return {"documents": rag.list_documents()}

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

@app.post("/send")
def receive_message(data: ChatRequest):
    user_msg = data.message
    document = data.document
    mode = data.mode

    if mode == "qa":
        result = qa_module.invoke(question=user_msg, document=document)
        clean_answer = result["Answer"][12:-2]  # preserve your old logic
        reply = format_qa_output(clean_answer)

    elif mode == "summary":
        summary_mode = data.summary_mode
        print(f"Invoking summary with mode: {summary_mode}")
        result = summary_module.invoke(
            question=user_msg,
            document=document,
            summary_mode=summary_mode
        )
        clean_answer = result['Answer']
        reply = format_summarize_output(clean_answer)

    elif mode == "slide_generation":
        reply = slide_generation_module(user_msg, document=document)
        save_as_pptx(reply, "layouts.pptx", "generated_slides.pptx")
        reply = "Slide generation completed and saved as 'generated_slides.pptx'."

    elif mode == "visualization":
        reply = visualization_module()

    else:
        reply = f"You said: {user_msg}"

    return {"reply": reply}

# ----------------------------
# Formatting Utilities
# ----------------------------
import re

def format_qa_output(raw_text: str) -> str:
    text = raw_text.strip()
    text = text.replace("\\n", "\n")
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = re.sub(r'\*+', '', text)
    text = re.sub(r'\s*(\d+)\.\s+', r'\n\1. ', text)
    text = re.sub(r'\s*\((\d+)\)\s*', r'\n(\1) ', text)
    text = re.sub(r'\s-\s+(?=[A-Z])', r'\n• ', text)
    text = re.sub(r'\s*•\s+', r'\n• ', text)
    text = re.sub(r'\n{2,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    lines = [line.strip() for line in text.split('\n')]
    return "\n".join(line for line in lines if line).strip()

def format_summarize_output(raw_text: str) -> str:
    answer = raw_text.strip()
    answer = answer.replace("\\n", " ")
    answer = re.sub(r'(\b\d)\.\s*\n\s*(\d)', r'\1.\2', answer)
    answer = re.sub(r'\s+', ' ', answer)
    sentences = re.split(r'(?<=[.!?]) +', answer)
    paragraphs = [" ".join(sentences[i:i+2]) for i in range(0, len(sentences), 2)]
    return "\n\n".join(paragraphs).strip()