from langchain.messages import SystemMessage, HumanMessage, AIMessage
from smart_doc.core.models import text_model as model
import smart_doc.features.summarization.prompts as prompts
import json
from smart_doc.utils.helper import safe_json_parse

MAX_TEXT_CHARS = 4000


def synthesis_agent(state: dict, model=model):
    """Merge text and image summaries into the final structured answer."""
    mode = state.get("summary_mode", "overview").upper()
    budget = state.get("token_budget", 300)
    detail = state.get("detail_level", 2)

    system_prompt = prompts.SA_SYSTEM_PROMPT.format(
        mode=mode,
        budget=budget,
        detail=detail
    )

    # Keep raw retrieved context bounded; modality summaries carry the full signal.
    text_combined = "\n".join(state.get("retrieved_text_chunks", []))
    if len(text_combined) > MAX_TEXT_CHARS:
        text_combined = text_combined[:MAX_TEXT_CHARS] + "..."

    payload = {
        "question": state.get("user_question", ""),
        "mode": mode,
        "token_budget": budget,
        "detail_level": detail,
        "text_chunks": text_combined,
        "images": state.get("image_captions", []),
        "text_summary": state.get("text_summary", ""),
        "image_summary": state.get("image_summary", "")
    }

    agent_answer: AIMessage = model.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=json.dumps(payload))
    ])

    parsed = safe_json_parse(
        agent_answer.content,
        {"Answer": agent_answer.content}
    )

    return {
        "final_summary": parsed,
        "llm_calls": 1
    }
