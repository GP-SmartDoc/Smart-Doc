from langchain.messages import SystemMessage, HumanMessage, AIMessage
from src.config.model import model
import src.config.visualization_prompts as prompts
import src.config.qa_prompts as qprompts
import json


def regenerating_agent(state: dict, model=model):

    system_prompt = prompts.REGENERATOR_SYSTEM_PROMPT

    if state.get('regenerator_output'):
        previous_code = state.get('regenerator_output')
    else:
        previous_code = state.get('generator_output')
        
    content = f"""
        Previous Code: {previous_code} 
        Revisor Insights: {state.get('revisor_output')}      
    """
    agent_answer: AIMessage = model.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=content)
    ])
    
    return {
        "messages": [agent_answer],
        "llm_calls": 1,
        "regenerator_ouput": agent_answer.content
    }
