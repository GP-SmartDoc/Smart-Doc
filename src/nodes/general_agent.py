from langchain.messages import SystemMessage, HumanMessage
from src.config.model import model
import src.config.prompts as prompts
import json


def general_agent(state: dict,model):

    resp = model.invoke([
        SystemMessage(content=prompts.GA_SYSTEM_PROMPT),
        HumanMessage(content=json.dumps({
            "question": state.get("user_question", ""),
            "retrieved_text_chunks": state.get("retrieved_text_chunks", []),
            "retrieved_images": state.get("retrieved_images", [])
        }))
    ])

    return {
        "general_context": resp.content,
        "llm_calls": 1
    }
