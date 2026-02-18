from langchain.messages import SystemMessage, HumanMessage, AIMessage
from config.model import visualization_model
import config.visualization_prompts as prompts
import config.qa_prompts as qprompts
import json


def regenerating_agent(state: dict):

    system_prompt = prompts.REGENERATOR_SYSTEM_PROMPT

    if state.get('regenerator_output'):
        previous_code = state.get('regenerator_output')
    else:
        previous_code = state.get('generator_output')
        
    content = f"""
        Previous Code: {previous_code} 
        Revisor Insights: {state.get('revisor_output')}      
    """
    agent_answer: AIMessage = visualization_model.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=content)
    ])
    print("REGENERATOR:", agent_answer.content)
    return {
        "messages": [agent_answer],
        "llm_calls": 1,
        "regenerator_ouput": agent_answer.content
    }
