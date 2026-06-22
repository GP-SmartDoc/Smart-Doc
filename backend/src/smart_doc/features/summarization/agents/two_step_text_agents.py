from langchain.messages import SystemMessage, HumanMessage
from smart_doc.core.models import text_model as model
import smart_doc.features.summarization.prompts as prompts
import json

MAX_TEXT_CHARS = 4000


def text_analyst_agent(state: dict, model=model):
    detail_level = state.get("detail_level", 2)

    all_text = state.get("retrieved_text_chunks", [])
    if not all_text:
        all_text = ["No relevant text found."]

    # Keep input size stable while preserving at least the beginning of the first chunk.
    truncated_text = []
    total_chars = 0

    for i, chunk in enumerate(all_text):
        if total_chars + len(chunk) > MAX_TEXT_CHARS:
            if i == 0:
                truncated_text.append(chunk[:MAX_TEXT_CHARS] + "...")
            break
        truncated_text.append(chunk)
        total_chars += len(chunk)

    if not truncated_text and all_text:
        truncated_text = [all_text[0][:MAX_TEXT_CHARS] + "..."]

    merged_text = "\n\n".join(truncated_text)

    resp = model.invoke([
        SystemMessage(content=prompts.TA_SYSTEM_PROMPT),
        HumanMessage(content=f"""
            Question:
            {state.get("user_question", "")}

            Text Chunks:
            {merged_text}
            
            Detail Level: {detail_level}
        """)
    ])

    return {
        "text_chunk_summaries": [resp.content],
        "llm_calls": 1
    }


def text_aggregator_agent(state: dict, model=model):
    payload = {
        "question": state.get("user_question", ""),
        "detail_level": state.get("detail_level", 2),
        "summaries": state.get("text_chunk_summaries", [])
    }

    resp = model.invoke([
        SystemMessage(content=prompts.TA_MODALITY_SYSTEM_PROMPT),
        HumanMessage(content=json.dumps(payload))
    ])

    return {
        "text_summary": resp.content,
        "llm_calls": 1
    }
