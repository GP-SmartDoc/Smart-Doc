from langchain.messages import SystemMessage, HumanMessage
from src.config.model import model
import src.config.summarization_prompts as prompts
import json


def text_micro_agent(state: dict,model):
    summaries = []

    for chunk in state.get("retrieved_text_chunks", []):
        resp = model.invoke([
            SystemMessage(content=prompts.TA_SYSTEM_PROMPT),
            HumanMessage(content=f"""
            Question:
            {state.get("user_question", "")}

            Text Chunk:
            {chunk}
            """)
        ])
        summaries.append(resp.content)

    return {
        "text_chunk_summaries": summaries,
        "llm_calls": len(summaries)
    }


def text_modality_agent(state: dict,model):
    resp = model.invoke([
        SystemMessage(content=prompts.TA_SYSTEM_PROMPT),
        HumanMessage(content=json.dumps({
            "question": state.get("user_question", ""),
            "chunk_summaries": state.get("text_chunk_summaries", [])
        }))
    ])

    return {
        "text_summary": resp.content,
        "llm_calls": 1
    }