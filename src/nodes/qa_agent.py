from langchain.messages import SystemMessage, HumanMessage, AIMessage
import src.config.qa_prompts as qa_prompts
import json

def qa_agent(state: dict, model):
    intent = state.get("intent", "qa")
    system_prompt = qa_prompts.QA_FINAL_SYSTEM_PROMPT 

    agent_answer: AIMessage = model.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=json.dumps(state.get("cross_modal_analysis", {})))
    ])

    return {
        "final_answer": agent_answer.content,
        "llm_calls": 1
    }

