from langchain.messages import SystemMessage, HumanMessage, AIMessage
from smart_doc.core.models import visualization_model
import smart_doc.features.visualization.prompts as prompts
import smart_doc.features.question_answering.prompts as qprompts
import json


def generating_agent(state: dict):
    
    d_type = state.get("diagram_type")
    current_syntax = prompts.SYNTAX_GUIDES.get(d_type)
    system_prompt = prompts.GENERATOR_SYSTEM_PROMPT.format(
        type=d_type,
        syntax=current_syntax
    )

    content = f"""
        Visualization Description: {state.get('description','')}
    """
    agent_answer: AIMessage = visualization_model.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=content)
    ])
    print("GENERATOR METADATA" ,agent_answer.response_metadata)
    print("GENERATOR", agent_answer.content)
    return {
        "messages": [agent_answer],
        "llm_calls": 1,
        "generator_output": agent_answer.content
    }
