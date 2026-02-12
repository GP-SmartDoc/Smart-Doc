from langchain.messages import SystemMessage, HumanMessage, AIMessage
from src.config.model import model
import src.config.visualization_prompts as prompts
import src.config.qa_prompts as qprompts
import json


def generating_agent(state: dict, model=model):
    
    system_prompt = prompts.GENERATOR_SYSTEM_PROMPT 

    content = f"""
        Visualization Description: {state.get('description','')}
    """
    agent_answer: AIMessage = model.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=content)
    ])
    
    return {
        "messages": [agent_answer],
        "llm_calls": 1,
        "generator_output": agent_answer.content
    }
