from langchain.messages import SystemMessage, HumanMessage
from src.config.model import model
import src.config.prompts as prompts
import json


def image_micro_agent(state: dict,model):
    image_summaries = []

    for image_path in state.get("retrieved_images", []):
        resp = model.invoke([
            SystemMessage(content=prompts.IA_SYSTEM_PROMPT),
            HumanMessage(content=f"""
            Question:
            {state.get("user_question", "")}

            Image Path:
            {image_path}
            """)
        ])
        image_summaries.append(resp.content)

    return {
        "image_summaries": image_summaries,
        "llm_calls": len(image_summaries)
    }


def image_modality_agent(state: dict,model):
    resp = model.invoke([
        SystemMessage(content=prompts.IA_SYSTEM_PROMPT),
        HumanMessage(content=json.dumps({
            "question": state.get("user_question", ""),
            "image_summaries": state.get("image_summaries", [])
        }))
    ])

    return {
        "image_summary": resp.content,
        "llm_calls": 1
    }
