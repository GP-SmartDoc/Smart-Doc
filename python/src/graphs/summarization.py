from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict, Annotated
import operator
from src.nodes.summarization.two_step_text_agents import text_micro_agent, text_modality_agent
from src.nodes.summarization.two_step_image_agents import image_micro_agent, image_modality_agent
from src.nodes.summarization.critical_agent import critical_agent
from src.nodes.summarization.summarizing_agent import summarization_agent
from src.states.summarization_state import SummarizerState
from src.vector_store.chroma import rag

g = StateGraph(SummarizerState)

g.add_node("text_micro", text_micro_agent)
g.add_node("image_micro", image_micro_agent)
g.add_node("text_merge", text_modality_agent)
g.add_node("image_merge", image_modality_agent)
g.add_node("critical", critical_agent)
g.add_node("final", summarization_agent)

g.add_edge(START, "text_micro")
g.add_edge(START, "image_micro")
g.add_edge("text_micro", "text_merge")
g.add_edge("image_micro", "image_merge")
g.add_edge("text_merge", "critical")
g.add_edge("image_merge", "critical")
g.add_edge("critical", "final")
g.add_edge("final", END)

summarization_module = g.compile()

def invoke(prompt):
    retrieved = rag.query(prompt, k_text=6, k_image=4)

    state = {
        "llm_calls": 0,
        "intent": "summary",
        "user_question": prompt,
        "retrieved_text_chunks": retrieved.get("text", []),
        "retrieved_images": retrieved.get("images", []),
        "text_chunk_summaries": [],
        "image_summaries": [],
        "text_summary": "",
        "image_summary": "",
        "cross_modal_analysis": {},
        "final_summary": {},
        "general_context": "",
    }
    result = summarization_module.invoke(state)
    return result["final_summary"]
