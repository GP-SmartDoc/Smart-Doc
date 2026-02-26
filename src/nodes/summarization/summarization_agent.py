from langchain.messages import SystemMessage, HumanMessage, AIMessage
from config.model import model
import config.summarization_prompts as prompts
import json
from utils.helper import safe_json_parse

MAX_TEXT_CHARS = 4000  # max characters to send to model


def summarization_agent(state: dict, model=model):
    """
    Final synthesis agent.
    Now supports controllable summary length via mode + token budget.
    """

    # ---------------- Mode Parameters ----------------
    mode = state.get("summary_mode", "overview").upper()
    budget = state.get("token_budget", 300)
    detail = state.get("detail_level", 2)

    system_prompt = prompts.SA_SYSTEM_PROMPT.format(
        mode=mode,
        budget=budget,
        detail=detail
    )

    # ---------------- Safe text ----------------
    text_combined = "\n".join(state.get("retrieved_text_chunks", []))
    if len(text_combined) > MAX_TEXT_CHARS:
        text_combined = text_combined[:MAX_TEXT_CHARS] + "…"

    # ---------------- Safe images ----------------
    images = state.get("image_captions", [])

    # ---------------- Safe cross-modal analysis ----------------
    cross_modal = state.get("cross_modal_analysis", {})
    cross_modal_safe = {
        k: cross_modal[k]
        for k in ["text", "image"]
        if k in cross_modal
    }

    # ---------------- Payload ----------------
    payload = {
        "question": state.get("user_question", ""),
        "mode": mode,
        "token_budget": budget,
        "detail_level": detail,
        "text_chunks": text_combined,
        "images": images,
        "cross_modal_analysis": cross_modal_safe
    }

    # ---------------- Invoke model ----------------
    agent_answer: AIMessage = model.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=json.dumps(payload))
    ])

    # ---------------- Parse output safely ----------------
    parsed = safe_json_parse(
        agent_answer.content,
        {"Answer": agent_answer.content}
    )

    return {
        "final_summary": parsed,
        "llm_calls": state.get("llm_calls", 0) + 1
    }