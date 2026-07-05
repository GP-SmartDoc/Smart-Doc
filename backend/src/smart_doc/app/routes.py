import os
from typing import List

from fastapi import APIRouter, File, Request, UploadFile, Header
from fastapi.responses import HTMLResponse
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
from smart_doc.core.rag_engine_proxy import RAGEngineProxy
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

rag = RAGEngineProxy()
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


# @router.get("/", response_class=HTMLResponse)
# def home(request: Request):
#     return templates.TemplateResponse(
#         "index.html",
#         {"request": request},
#     )


@router.get("/documents")
def list_documents(x_user_id: str | None = Header(default=None)):
    return {"documents": rag.list_documents(user_id=x_user_id)}


@router.post("/send")
def receive_message(data: ChatRequest, x_user_id: str | None = Header(default=None)):
    user_msg = data.message
    document = data.document
    enhanced_question = user_msg

    if data.mode == "qa":
        result = qa_module.invoke(
            question=enhanced_question,
            document=document,
            user_id=x_user_id,
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
            user_id=x_user_id,
        )

        clean_answer = extract_answer(result)
        if(clean_answer[0] == '{'):
            clean_answer = clean_answer[12:-2]

        reply = formatting.format_summarize_output(clean_answer)

    elif data.mode == "slide_generation":
        reply = generate_slides(
            rag,
            enhanced_question,
            document=document,
            user_id=x_user_id,
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
            user_id=x_user_id,
        )

    else:
        reply = f"You said: {user_msg}"

    return {"reply": reply}


@router.post("/upload")
async def upload_files(files: List[UploadFile] = File(...), x_user_id: str | None = Header(default=None)):
    uploaded_files = []
    skipped_files = []
    failed_files = []

    for file in files:
        user_upload_folder = os.path.join(UPLOAD_FOLDER, x_user_id) if x_user_id else UPLOAD_FOLDER
        os.makedirs(user_upload_folder, exist_ok=True)
        file_path = os.path.join(user_upload_folder, file.filename)
        file_existed = os.path.exists(file_path)

        try:
            with open(file_path, "wb") as f:
                f.write(await file.read())

            result = rag.add_file(file_path, user_id=x_user_id)
            print(f"File ingestion result for {file.filename}: {result}")
            if result.get("status") == "skipped":
                # The file is a duplicate in the vector DB (same content was indexed before,
                # possibly by another user). The file is intentionally kept in this user's
                # folder so it shows up in their documents list and they can query it.
                # From the user's perspective this is a successful upload.
                uploaded_files.append({"filename": file.filename, "task_id": None})
            elif result.get("status") == "queued":
                uploaded_files.append({"filename": file.filename, "task_id": result.get("task_id")})
            else:
                uploaded_files.append(file.filename)

        except Exception as e:
            print(f"Error handling {file.filename}: {e}")
            if not file_existed and os.path.exists(file_path):
                os.remove(file_path)
            failed_files.append(file.filename)

    return {
        "uploaded": uploaded_files,
        "skipped": skipped_files,
        "failed": failed_files
    }