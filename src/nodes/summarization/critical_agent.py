from langchain.messages import SystemMessage, HumanMessage
from config.model import model
import config.summarization_prompts as prompts
import config.qa_prompts as qprompts
import json


def critical_agent(state: dict, model):
    # ---------- FIX: do not overwrite valid analysis ----------
    if state.get("cross_modal_analysis"):
        return {"llm_calls": 0}

    payload = {
        "question": state.get("user_question", ""),
        "text": state.get("text_answer", ""),
        "image": state.get("image_answer", "")
    }

    resp = model.invoke([
        SystemMessage(content=qprompts.QA_CA_SYSTEM_PROMPT),
        HumanMessage(content=json.dumps(payload))
    ])

    try:
        analysis = json.loads(resp.content)
    except Exception:
        analysis = {
            "text": state.get("text_answer", ""),
            "image": state.get("image_answer", "")
        }

    print("[DEBUG] Critical analysis:", analysis)

    return {
        "cross_modal_analysis": analysis,
        "llm_calls": 1
    }


