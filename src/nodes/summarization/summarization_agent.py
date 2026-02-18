from langchain.messages import SystemMessage, HumanMessage, AIMessage
from config.model import model
import config.summarization_prompts as prompts
import json
from utils.helper import safe_json_parse

MAX_TEXT_CHARS = 4000  # max characters to send to model

def summarization_agent(state: dict, model=model):
    """
    Combines retrieved text and images into a final summary safely.
    Truncates text and images to avoid API size limits.
    """
    system_prompt = prompts.SA_SYSTEM_PROMPT

    # ---------------- Safe text ----------------
    text_combined = "\n".join(state.get("retrieved_text_chunks", []))
    if len(text_combined) > MAX_TEXT_CHARS:
        text_combined = text_combined[:MAX_TEXT_CHARS] + "…"

    # ---------------- Safe images ----------------
    images = state.get("image_captions", [])

    # ---------------- Safe cross-modal analysis ----------------
    cross_modal = state.get("cross_modal_analysis", {})
    cross_modal_safe = {k: cross_modal[k] for k in ["text", "image"] if k in cross_modal}

    payload = {
        "question": state.get("user_question", ""),
        "text_chunks": text_combined,
        "images": images,
        "cross_modal_analysis": cross_modal_safe
    }

    # ---------------- Invoke model ----------------
    agent_answer: AIMessage = model.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=json.dumps(payload))
    ])

    return {
        "final_summary": safe_json_parse(
            agent_answer.content,
            {"summary": agent_answer.content}
        ),
        "llm_calls": 1
    }
