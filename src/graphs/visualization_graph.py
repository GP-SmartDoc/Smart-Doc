import os
import shutil
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain.messages import AnyMessage, HumanMessage, AIMessage

from nodes.visualization.generating_agent import generating_agent
from nodes.visualization.regenerating_agent import regenerating_agent
from nodes.visualization.revising_agent import revising_agent
from states.visualization_state import VisualizationGraphState, DiagramType

from utils.strings import remove_thinking_from_content

__all__ = ["generate_visualization"]

def should_continue(state: VisualizationGraphState):
    if state["done"]:
        return END  
    else:
        return "regenerator"  
    
builder = StateGraph(VisualizationGraphState)

builder.add_node("generator", generating_agent)
builder.add_node("revisor", revising_agent)
builder.add_node("regenerator", regenerating_agent)

builder.add_edge(START, "generator")
builder.add_edge("generator", "revisor")
builder.add_conditional_edges("revisor", should_continue)
builder.add_edge("regenerator", "revisor")

visualization_module = builder.compile()

def generate_visualization(type:DiagramType, description:str)->str:

    initial_state = {
        "messages": [HumanMessage(content=description)],
        "llm_calls": 0,
        
        "description": description,
        "diagram_type": type,
        
        "generator_output" : "",
        "revisor_output" : "",
        "regenerator_output" : "",
        "done" : False
    }
    
    final_state = visualization_module.invoke(initial_state)
    if final_state.get("regenerator_output") != "":
        return remove_thinking_from_content(final_state.get("regenerator_output"))
    else:
        return remove_thinking_from_content(final_state.get("generator_output"))