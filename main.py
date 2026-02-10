# main.py
import os
import sys
import chromadb
from src.config.model import model
from src.vector_store.RAG import RAGEngine
from src.graphs.summary_graph import SummarizationModule
from src.graphs.qa_graph import QuestionAnsweringModule
from src.graphs.visualize_graph import generate_slides
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
    Returns one of: 'summary', 'visualization', 'qa'.
    """
    prompt = f"""
Determine the intent of the following user input.
Return ONLY one word: 'summary', 'visualization', or 'qa'.

User Input:
\"\"\"{user_input}\"\"\"
"""
    resp = model.invoke([
        SystemMessage(content="You are an assistant that classifies user intent."),
        HumanMessage(content=prompt)
    ])
    
    # Normalize response
    intent = resp.content.strip().lower()
    if intent not in ["summary", "visualization", "qa"]:
        return "qa"  # fallback default
    return intent




# ----------------------------
# GUI
# ----------------------------
app = FastAPI()
templates = Jinja2Templates(directory="templates")

UPLOAD_FOLDER = "uploads"
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
    visualization_module = lambda prompt: generate_slides(rag, prompt)
    print("System Ready.\n")
except Exception as e:
    print(f"Initialization Error: {e}")

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/send")
def receive_message(data: dict):
    user_msg = data.get("message", "")
    mode = data.get("mode", "qa")  # default to "qa"

    ##########################################
    """#########  Call LLM Here Based on Mode ##########"""
    ##########################################

    if mode == "qa":
        result = qa_module.invoke(user_msg)
        print("##########################################")
        print("QA TYPE:", type(result), result)
        reply = result["Answer"][12:-2]

    elif mode == "summarize":
        result = summary_module.invoke(user_msg)
        reply = result['Answer']
       
    elif mode == "viz":
        reply = visualization_module(user_msg)
        save_as_pptx(reply, "generated_slides.pptx")
        reply = "Visualization generated and saved as 'generated_slides.pptx'."
    else:
        reply = f"You said: {user_msg}"

    return {"reply": reply}



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