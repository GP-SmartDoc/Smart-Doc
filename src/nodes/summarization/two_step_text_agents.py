from langchain.messages import SystemMessage, HumanMessage
from config.model import model
import config.summarization_prompts as prompts
import config.qa_prompts as qprompts
import json


def text_micro_agent(state: dict, model):
    intent = state.get("intent", "qa")

    # Aggregate all text chunks into a single string
    all_text = "\n\n".join(state.get("retrieved_text_chunks", []))
    if not all_text:
        all_text = "No relevant text found."

    system_prompt = (
        prompts.TA_SYSTEM_PROMPT if intent == "summary" else qprompts.QA_TA_SYSTEM_PROMPT
    )

    resp = model.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"""
            Question:
            {state.get("user_question", "")}

            Text Chunks:
            {all_text}
        """
        )
    ])

    key = "text_chunk_summaries" if intent == "summary" else "text_answer"
    return {
        key: resp.content,
        "llm_calls": 1
    }


def text_modality_agent(state: dict, model):
    intent = state.get("intent", "qa")

    if intent == "summary":
        system_prompt = prompts.TA_MODALITY_SYSTEM_PROMPT
        payload = {
            "question": state.get("user_question", ""),
            "summaries": state.get("text_chunk_summaries", [])
        }
        out_key = "text_summary"
    else:
        system_prompt = qprompts.QA_GA_SYSTEM_PROMPT
        payload = {
            "question": state.get("user_question", ""),
            "text_evidence": state.get("text_evidence", [])
        }
        out_key = "text_answer"

    resp = model.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=json.dumps(payload))
    ])
    print("[DEBUG] Text summary:", resp.content)
    return {
        out_key: resp.content,
        "llm_calls": 1
    }
