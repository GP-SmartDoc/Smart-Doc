from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict, Annotated
import operator
from smart_doc.features.summarization.agents.summarization_agent import synthesis_agent
from smart_doc.features.summarization.agents.two_step_text_agents import text_analyst_agent, text_aggregator_agent
from smart_doc.features.summarization.agents.two_step_image_agents import image_analyst_agent, image_aggregator_agent
from smart_doc.utils.compression_budget import compute_budget
from smart_doc.features.summarization.summary_modes import SummaryMode, MODE_CONFIG


class SummarizerState(TypedDict, total=False):
    llm_calls: Annotated[int, operator.add]
    user_question: str
    retrieved_text_chunks: list
    image_captions: list
    text_summary: str
    image_summary: str
    text_chunk_summaries: list
    image_answers: list
    summary_mode: str
    token_budget: int
    detail_level: int

    final_summary: dict


class SummarizationModule:
    def __init__(self, retriever):
        self.retriever = retriever

        g = StateGraph(SummarizerState)

        g.add_node("text_analyst", text_analyst_agent)
        g.add_node("image_analyst", image_analyst_agent)
        g.add_node("text_aggregator", text_aggregator_agent)
        g.add_node("image_aggregator", image_aggregator_agent)
        g.add_node("synthesis", synthesis_agent)

        # Text and image evidence are summarized independently, then merged once.
        g.add_edge(START, "text_analyst")
        g.add_edge(START, "image_analyst")
        g.add_edge("text_analyst", "text_aggregator")
        g.add_edge("image_analyst", "image_aggregator")
        g.add_edge("text_aggregator", "synthesis")
        g.add_edge("image_aggregator", "synthesis")
        g.add_edge("synthesis", END)

        self.app = g.compile()

    def invoke(self, question: str, document: str = "all", summary_mode: str = "overview", user_id: str = None):
        """
        Invokes the summarization pipeline.
        summary_mode: one of 'snapshot', 'overview', 'deepdive'
        """
        try:
            mode_enum = SummaryMode(summary_mode)
        except ValueError:
            mode_enum = SummaryMode.OVERVIEW

        # Intercept generic queries to pull better context from the vector store
        search_query = question
        if len(question.split()) <= 5 and any(w in question.lower() for w in ["summary", "brief", "detail"]):
            search_query = "abstract introduction main contribution methodology conclusion"

        retrieved = self.retriever.query(search_query, k_text=6, k_image=4, document=document, user_id=user_id)

        text = " ".join(retrieved.get("text", []))
        doc_tokens = max(1, len(text) // 4)
        config = MODE_CONFIG[mode_enum]
        budget = compute_budget(doc_tokens, config)

        if mode_enum != SummaryMode.SNAPSHOT:
            budget = max(budget, 60)

        detail = config["detail"]

        state = {
            "llm_calls": 0,
            "user_question": question,
            "retrieved_text_chunks": retrieved.get("text", []),
            "image_captions": retrieved.get("captions", []),
            "text_chunk_summaries": [],
            "image_answers": [],
            "text_summary": "",
            "image_summary": "",
            "final_summary": {},
            "summary_mode": mode_enum.value,
            "token_budget": budget,
            "detail_level": detail,
        }

        result = self.app.invoke(state)
        return result["final_summary"]
