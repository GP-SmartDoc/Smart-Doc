from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict, Annotated
import operator
from src.nodes.summarization.summarization_agent import summarization_agent
from src.nodes.summarization.critical_agent import critical_agent
from src.nodes.summarization.two_step_text_agents import text_micro_agent, text_modality_agent
from src.nodes.summarization.two_step_image_agents import image_micro_agent, image_modality_agent
from src.utils.compression_budget import compute_budget
from src.config.summary_modes import SummaryMode, MODE_CONFIG

# =========================
# STATE
# =========================
class SummarizerState(TypedDict, total=False):
    llm_calls: Annotated[int, operator.add]  # <-- fix here
    intent: str
    user_question: str
    retrieved_text_chunks: list
    retrieved_images: list
    image_captions: list
    text_summary: str
    image_summary: str
    cross_modal_analysis: dict

    text_chunk_summaries: list
    image_answers: list
    summary_mode: str
    token_budget: int
    detail_level: int

    final_summary: dict


# =========================
# GRAPH MODULE
# =========================
class SummarizationModule:
    def __init__(self, retriever, model):
        self.retriever = retriever
        self.model = model

        g = StateGraph(SummarizerState)

        # ---------- Nodes ----------
        g.add_node("text_micro", lambda s: text_micro_agent(s, self.model))
        g.add_node("image_micro", lambda s: image_micro_agent(s, self.model))
        g.add_node("text_merge", lambda s: text_modality_agent(s, self.model))
        g.add_node("image_merge", lambda s: image_modality_agent(s, self.model))
        g.add_node("critical", lambda s: critical_agent(s, self.model))
        g.add_node("final", lambda s: summarization_agent(s, self.model))

        # ---------- Edges ----------
        g.add_edge(START, "text_micro")
        g.add_edge(START, "image_micro")
        g.add_edge("text_micro", "text_merge")
        g.add_edge("image_micro", "image_merge")
        g.add_edge("text_merge", "critical")
        g.add_edge("image_merge", "critical")
        g.add_edge("critical", "final")
        g.add_edge("final", END)

        self.app = g.compile()

    def invoke(self, question: str, document: str = "all", summary_mode: str = "overview"):
        """
        Invokes the summarization pipeline.
        summary_mode: one of 'snapshot', 'overview', 'deepdive'
        """
        # Validate summary mode
        try:
            mode_enum = SummaryMode(summary_mode)
        except ValueError:
            mode_enum = SummaryMode.OVERVIEW

        # Intercept generic queries to pull better context from the vector store
        search_query = question
        if len(question.split()) <= 5 and any(w in question.lower() for w in ["summary", "brief", "detail"]):
            search_query = "abstract introduction main contribution methodology conclusion"

        # Retrieve relevant content
        retrieved = self.retriever.query(search_query, k_text=6, k_image=4, document=document)

        # Compute token budget
        text = " ".join(retrieved.get("text", []))
        doc_tokens = max(1, len(text) // 4)
        config = MODE_CONFIG[mode_enum]
        budget = compute_budget(doc_tokens, config)

        # Ensure minimum budget for non-snapshot modes
        if mode_enum != SummaryMode.SNAPSHOT:
            budget = max(budget, 60)

        detail = config["detail"]

        # Build initial state
        state = {
            "llm_calls": 0,
            "intent": "summary",
            "user_question": question,
            "retrieved_text_chunks": retrieved.get("text", []),
            "retrieved_images": retrieved.get("images", []),
            "image_captions": retrieved.get("captions", []),
            "text_chunk_summaries": [],
            "image_answers": [],
            "text_summary": "",
            "image_summary": "",
            "cross_modal_analysis": {},
            "final_summary": {},
            "summary_mode": mode_enum.value,
            "token_budget": budget,
            "detail_level": detail,
        }

        # Run the graph
        result = self.app.invoke(state)
        return result["final_summary"]