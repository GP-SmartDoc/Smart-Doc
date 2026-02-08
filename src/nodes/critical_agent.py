from langchain.messages import SystemMessage, HumanMessage
from src.config.model import model
import src.config.prompts as prompts
import json


def critical_agent(state: dict,model):
    resp = model.invoke([
        SystemMessage(content=prompts.CA_SYSTEM_PROMPT),
        HumanMessage(content=json.dumps({
            "question": state.get("user_question", ""),
            "text_summary": state.get("text_summary", ""),
            "image_summary": state.get("image_summary", ""),
            "general_context": state.get("general_context", "")
        }))
    ])

    return {
        "critical_analysis": resp.content,
        "llm_calls": 1
    }
