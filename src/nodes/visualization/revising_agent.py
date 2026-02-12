from langchain.messages import SystemMessage, HumanMessage, AIMessage
from src.config.model import model
import src.config.visualization_prompts as prompts
import src.config.qa_prompts as qprompts
import json


def revising_agent(state: dict, model=model):

    system_prompt = prompts.REVISOR_SYSTEM_PROMPT

    content = f"""
        Code: {state.get("generator_output")}        
    """
    agent_answer: AIMessage = model.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=content)
    ])
    
    if agent_answer.content == "ok":
        return {
            "messages": [agent_answer],
            "llm_calls": 1,
            "done": True
        }
    else:
        return {
            "messages": [agent_answer],
            "llm_calls": 1,
            "revisor_ouput": agent_answer.content
        }

