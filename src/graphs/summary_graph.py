from click import prompt
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict, Annotated
import operator
from nodes.summarization.summarization_agent import summarization_agent
from nodes.summarization.critical_agent import critical_agent
from nodes.summarization.two_step_text_agents import text_micro_agent, text_modality_agent
from nodes.summarization.two_step_image_agents import image_micro_agent, image_modality_agent


class SummarizerState(TypedDict):
    llm_calls: Annotated[int, operator.add]
    intent: str
    user_question: str
    retrieved_text_chunks: list
    retrieved_images: list
    image_captions: list
    text_summary: str
    image_summary: str
    cross_modal_analysis: dict
    final_summary: dict

class SummarizationModule:
    """
    A simple summarization module that uses only the summarization agent.
    """
    def __init__(self, retriever, model):
        self.retriever = retriever
        self.model = model

        g = StateGraph(SummarizerState)

        g.add_node("text_micro", lambda s: text_micro_agent(s, self.model))
        g.add_node("image_micro", lambda s: image_micro_agent(s, self.model))
        g.add_node("text_merge", lambda s: text_modality_agent(s, self.model))
        g.add_node("image_merge", lambda s: image_modality_agent(s, self.model))
        g.add_node("critical", lambda s: critical_agent(s, self.model))
        g.add_node("final", lambda s: summarization_agent(s, self.model))

        g.add_edge(START, "text_micro")
        g.add_edge(START, "image_micro")
        g.add_edge("text_micro", "text_merge")
        g.add_edge("image_micro", "image_merge")
        g.add_edge("text_merge", "critical")
        g.add_edge("image_merge", "critical")
        g.add_edge("critical", "final")
        g.add_edge("final", END)

        self.app = g.compile()

    def invoke(self, question: str):
        retrieved = self.retriever.query(question, k_text=6, k_image=4)

        state = {
        "llm_calls": 0,
        "intent": "summary",
        "user_question": question,
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
        result = self.app.invoke(state)
        return result["final_summary"]
