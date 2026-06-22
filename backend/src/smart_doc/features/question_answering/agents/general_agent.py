from langchain.messages import SystemMessage, HumanMessage, AIMessage
from smart_doc.core.models import text_model as model
import smart_doc.features.summarization.prompts as prompts
import smart_doc.features.question_answering.prompts as qprompts
import json


def general_agent(state: dict, model=model):
    intent = state.get("intent", "qa")
    
    system_prompt = prompts.GA_SYSTEM_PROMPT if intent == "summary" else qprompts.QA_GA_SYSTEM_PROMPT

    content = f"""
        Question: {state.get('user_question','')}
        Textual Content: {state.get('retrieved_text_chunks', [])}
    """
    agent_answer: AIMessage = model.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=content)
    ])
    
    return {
        "messages": [agent_answer],
        "llm_calls": 1,
        "ga_output": agent_answer.content
    }
