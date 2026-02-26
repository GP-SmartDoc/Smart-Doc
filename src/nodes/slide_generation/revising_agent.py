from langchain.messages import SystemMessage, HumanMessage, AIMessage
from config.model import visualization_model
import config.visualization_prompts as prompts
import config.qa_prompts as qprompts

from utils.strings import remove_thinking_from_content

def revising_agent(state: dict):
    
    d_type = state.get("diagram_type")
    current_syntax = prompts.SYNTAX_GUIDES.get(d_type)
    system_prompt = prompts.REVISOR_SYSTEM_PROMPT.format(
        type=d_type,
        syntax=current_syntax
    )
    
    if state.get("regenerator_output") != "":
        content = f"""
            Description: {state.get("description")}
            Code: {state.get("regenerator_output")}        
        """
    else:
        content = f"""
            Description: {state.get("description")}
            Code: {state.get("generator_output")}        
        """

    agent_answer: AIMessage = visualization_model.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=content)
    ])
    
    print("REVISOR: ", agent_answer.content)
    if remove_thinking_from_content(agent_answer.content) == "ok":
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

