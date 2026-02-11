from langchain.messages import SystemMessage, HumanMessage, AIMessage
from src.config.model import model
import src.config.summarization_prompts as prompts
import src.config.qa_prompts as qprompts
import json


def critical_agent(state: dict, model=model):
    intent = state.get("intent", "qa")
    system_prompt = prompts.CA_SYSTEM_PROMPT if intent == "summary" else qprompts.QA_CA_SYSTEM_PROMPT

    payload = {
        "question": state.get("user_question", ""),
        "text": state.get("text_answer", "") or state.get("text_summary", ""),
        "image": state.get("image_answer", "") or state.get("image_summary", "")
    }

    agent_answer: AIMessage = model.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=json.dumps(payload))
    ])

    try:
        analysis = json.loads(agent_answer.content)
    except Exception:
        analysis = payload

    return {
        "messages": [agent_answer],
        "llm_calls": 1,
        "ca_output": agent_answer.content,
        "cross_modal_analysis": analysis
    }


