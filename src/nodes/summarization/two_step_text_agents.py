from langchain.messages import SystemMessage, HumanMessage
from config.model import model
import config.summarization_prompts as prompts
import config.qa_prompts as qprompts
import json

def text_micro_agent(state: dict, model):
    intent = state.get("intent", "qa")
    detail_level = state.get("detail_level", 2)  # from mode_controller
    
    # We retrieve the token_budget but do NOT use it to truncate the input text anymore.
    # The budget is handled strictly by the final synthesis agent's prompt.
    token_budget = state.get("token_budget", 300)

    all_text = state.get("retrieved_text_chunks", [])
    if not all_text:
        all_text = ["No relevant text found."]

    # ---------------------------
    # Decoupled Input Context
    # Give the LLM a generous 4000 characters to read, regardless of output size
    # ---------------------------
    max_chars = 4000 
    truncated_text = []
    total_chars = 0
    
    for i, chunk in enumerate(all_text):
        if total_chars + len(chunk) > max_chars:
            # Always include at least the first chunk even if over the max_chars limit
            if i == 0:
                truncated_text.append(chunk[: max_chars] + "…")
            break
        truncated_text.append(chunk)
        total_chars += len(chunk)

    # fallback if truncation leaves nothing
    if not truncated_text and all_text:
        truncated_text = [all_text[0][: max_chars] + "…"]

    merged_text = "\n\n".join(truncated_text)

    # Choose prompt based on intent
    system_prompt = prompts.TA_SYSTEM_PROMPT if intent == "summary" else qprompts.QA_TA_SYSTEM_PROMPT
    resp = model.invoke([
        SystemMessage(content=system_prompt),
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