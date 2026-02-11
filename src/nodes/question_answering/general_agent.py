from langchain.messages import SystemMessage, HumanMessage, AIMessage
from src.config.model import model
import src.config.summarization_prompts as prompts
import src.config.qa_prompts as qprompts
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
