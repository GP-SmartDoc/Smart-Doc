from langchain.messages import SystemMessage, HumanMessage
from src.config.model import model
import src.config.summarization_prompts as prompts
import src.config.qa_prompts as qprompts
import json


def general_agent(state: dict, model):
    intent = state.get("intent", "qa")

    if intent == "summary":
        system_prompt = prompts.GA_SYSTEM_PROMPT
        payload = {
            "question": state.get("user_question", ""),
            "retrieved_text_chunks": state.get("retrieved_text_chunks", []),
            "retrieved_images": state.get("retrieved_images", [])
        }
    else:  # QA
        system_prompt = qprompts.QA_GA_SYSTEM_PROMPT
        payload = {
            "question": state.get("user_question", ""),
            "retrieved_text_chunks": state.get("retrieved_text_chunks", [])
        }

    resp = model.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=json.dumps(payload))
    ])

    return {
        "general_context": resp.content,
        "llm_calls": 1
    }
