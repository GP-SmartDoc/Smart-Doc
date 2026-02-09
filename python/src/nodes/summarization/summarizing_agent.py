from langchain.messages import SystemMessage, HumanMessage
from src.config.model import model
import src.config.summarization_prompts as prompts
import json
from src.utils.json import safe_json_parse

def summarization_agent(state, model):
    resp = model.invoke([
        SystemMessage(content=prompts.SA_SYSTEM_PROMPT),
        HumanMessage(
            content=json.dumps({
                "question": state["user_question"],
                "text_summary": state["text_summary"],
                "image_summary": state["image_summary"],
                "analysis": state["cross_modal_analysis"]
            })
        )
    ])

    return {
        "final_summary": safe_json_parse(
            resp.content,
            {"summary": resp.content}
        ),
        "llm_calls": 1
    }