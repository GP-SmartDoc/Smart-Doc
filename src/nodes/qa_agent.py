from langchain.messages import SystemMessage, HumanMessage, AIMessage
import src.config.qa_prompts as qa_prompts
import json

def qa_agent(state: dict, model):
    resp = model.invoke([
        SystemMessage(content=qa_prompts.QA_FINAL_SYSTEM_PROMPT),
        HumanMessage(content=json.dumps(state.get("cross_modal_analysis", {})))
    ])

    return {
        "final_answer": resp.content,
        "llm_calls": 1
    }

