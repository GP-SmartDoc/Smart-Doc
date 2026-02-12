import os
import shutil
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain.messages import AnyMessage, HumanMessage, AIMessage

from src.nodes.visualization.generating_agent import generating_agent
from src.nodes.visualization.regenerating_agent import regenerating_agent
from src.nodes.visualization.revising_agent import revising_agent
from src.states.visualization_state import VisualizationGraphState


def should_continue(state: VisualizationGraphState):
    if state["ok"]:
        return END  
    else:
        return "generator"  
    
builder = StateGraph(VisualizationGraphState)

builder.add_node("generator", generating_agent)
builder.add_node("revisor", revising_agent)
builder.add_node("regenerator", regenerating_agent)

builder.add_edge(START, "generator")
builder.add_edge("generator", "regenerator")
builder.add_conditional_edges("regenerator", should_continue)

visualization_module = builder.compile()

def generate_visualization(rag,prompt:str)->str:

    initial_state = {
        "messages": [HumanMessage(content=prompt)],
        "llm_calls": 0,
        "description": prompt,
        
        "generator_output" : "",
        "revisor_output" : "",
        "regenerator_output" : "",
        "done" : False
    }
    
    final_state = visualization_module.invoke(initial_state)
    return final_state.get("regenerator_output", "")