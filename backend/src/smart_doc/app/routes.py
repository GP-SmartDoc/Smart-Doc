import os
from typing import List

import chromadb
from fastapi import APIRouter, File, Request, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates

from smart_doc.app import formatting
from smart_doc.app.schemas import ChatRequest
from smart_doc.app.settings import (
    BLOB_STORAGE_FOLDER,
    CHROMA_DB_FOLDER,
    OUTPUT_DIR,
    SLIDES_OUTPUT_PATH,
    UPLOAD_FOLDER,
)
from smart_doc.core.chat_memory import ChatMemory
from smart_doc.features.question_answering.graph import QuestionAnsweringModule
from smart_doc.features.slide_generation.graph import generate_slides
from smart_doc.features.summarization.graph import SummarizationModule
from smart_doc.features.visualization.rag_graph import VisualizationModule
from smart_doc.features.visualization.state import DiagramType
from smart_doc.retrieval.rag_engine import RAGEngine
from smart_doc.utils.helper import safe_json_parse
from smart_doc.utils.pptx import save_as_pptx


router = APIRouter()
templates = Jinja2Templates(directory="templates")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def detect_diagram_type(prompt: str) -> DiagramType:
    text = prompt.lower()

    if "sequence" in text:
        return DiagramType.SEQUENCE
    if "state" in text:
        return DiagramType.STATE
    if "class" in text:
        return DiagramType.CLASS
    if "er" in text or "entity relationship" in text:
        return DiagramType.ER
    if "pie" in text:
        return DiagramType.PIE
    if "mindmap" in text or "mind map" in text:
        return DiagramType.MINDMAP

    return DiagramType.FLOWCHART


print("--- Initializing Smart Doc System ---")

client = chromadb.PersistentClient(path=CHROMA_DB_FOLDER)
rag = RAGEngine(
    client,
    blob_storage_path=BLOB_STORAGE_FOLDER,
    documents_path=UPLOAD_FOLDER,
)
qa_module = QuestionAnsweringModule(retriever=rag)
summary_module = SummarizationModule(retriever=rag)
visualization_module = VisualizationModule(retriever=rag)
memory = ChatMemory()

print("System Ready.\n")


def extract_answer(result) -> str:
    parsed = safe_json_parse(result, {"Answer": str(result)})
    answer = parsed.get("Answer", str(result)) if isinstance(parsed, dict) else str(parsed)

    nested = safe_json_parse(answer, {})
    if isinstance(nested, dict) and "Answer" in nested:
        return nested["Answer"]

    return str(answer)


def build_summary_reply(result) -> str:
    if isinstance(result, str):
        parsed = safe_json_parse(result, {})
        if isinstance(parsed, dict):
            result = parsed

    answer_text = ""
    if isinstance(result, dict):
        answer_text = result.get("Answer", "") or result.get("final_summary", "")
        if isinstance(answer_text, dict):
            answer_text = str(answer_text)
    else:
        answer_text = str(result)

    summary_text = answer_text.strip()
    if "```mermaid" not in summary_text:
        summary_text = formatting.format_summarize_output(summary_text)

    if isinstance(result, dict):
        diagram = result.get("Diagram", "").strip()
        if diagram:
            # 🔧 FIX: Just add the mermaid diagram, without the reasoning text
            diagram_section = f"\n\n```mermaid\n{diagram}\n```"
            
            return summary_text + diagram_section

    return summary_text


# @router.get("/", response_class=HTMLResponse)
# def home(request: Request):
#     return templates.TemplateResponse(
#         "index.html",
#         {"request": request},
#     )


@router.get("/documents")
def list_documents():
    return {"documents": rag.list_documents()}


@router.post("/send")
def receive_message(data: ChatRequest):
    user_msg = data.message
    document = data.document
    enhanced_question = user_msg

    if data.mode == "qa":
        result = qa_module.invoke(
            question=enhanced_question,
            document=document,
        )
        clean_answer = extract_answer(result)
        if(clean_answer[0] == '{'):
            clean_answer = clean_answer[12:-2]
        reply = formatting.format_qa_output(clean_answer)

    elif data.mode == "summary":
        print(f"Invoking summary with mode: {data.summary_mode}")
        result = summary_module.invoke(
            question=enhanced_question,
            document=document,
            summary_mode=data.summary_mode,
        )

        reply = build_summary_reply(result)

    elif data.mode == "slide_generation":
        reply = generate_slides(
            rag,
            enhanced_question,
            document=document,
        )
        save_as_pptx(
            reply,
            "layouts.pptx",
            SLIDES_OUTPUT_PATH,
        )
        reply = f"Slide generation completed and saved as '{SLIDES_OUTPUT_PATH}'."

    elif data.mode == "visualization":
        reply = visualization_module.invoke(
            request=enhanced_question,
            diagram_type=detect_diagram_type(user_msg),
            document=document,
        )

    else:
        reply = f"You said: {user_msg}"

    return {"reply": reply}


@router.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    uploaded_files = []
    skipped_files = []
    failed_files = []

    for file in files:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file_existed = os.path.exists(file_path)

        try:
            with open(file_path, "wb") as f:
                f.write(await file.read())

            result = rag.add_file(file_path)
            print(f"File ingestion result for {file.filename}: {result}")
            if result.get("status") == "skipped":
                if not file_existed and os.path.exists(file_path):
                    os.remove(file_path)
                skipped_files.append(file.filename)
            else:
                uploaded_files.append(file.filename)

        except Exception as e:
            if not file_existed and os.path.exists(file_path):
                os.remove(file_path)
            print(f"Error uploading {file.filename}: {e}")
            failed_files.append(
                {
                    "file": file.filename,
                    "error": str(e),
                }
            )

    return {
        "uploaded": uploaded_files,
        "skipped": skipped_files,
        "failed": failed_files,
    }

@router.get("/download-slides")
async def download_slides():
    # Use the SLIDES_OUTPUT_PATH variable already imported in your file
    file_path = SLIDES_OUTPUT_PATH
    
    if os.path.exists(file_path):
        return FileResponse(
            path=file_path, 
            filename="generated_slides.pptx", 
            media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation"
        )
    else:
        raise HTTPException(status_code=404, detail="Slide file not found on the server.")