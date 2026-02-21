from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict, Annotated
import operator
from src.nodes.summarization.summarization_agent import summarization_agent
from src.nodes.summarization.critical_agent import critical_agent
from src.nodes.summarization.two_step_text_agents import text_micro_agent, text_modality_agent
from src.nodes.summarization.two_step_image_agents import image_micro_agent, image_modality_agent
from src.utils.compression_budget import compute_budget
from src.utils.summary_classifier import classify_mode
from src.config.summary_modes import SummaryMode, MODE_CONFIG

# =========================
# STATE
# =========================
class SummarizerState(TypedDict, total=False):
    llm_calls: Annotated[int, operator.add]
    intent: str
    user_question: str
    retrieved_text_chunks: list
    retrieved_images: list
    image_captions: list
    text_summary: str
    image_summary: str
    cross_modal_analysis: dict

    # ---- MISSING KEYS ADDED HERE ----
    text_chunk_summaries: list
    image_answers: list
    summary_mode: str
    token_budget: int
    detail_level: int

    # ---- FINAL ----
    final_summary: dict

# =========================
# MODE CONTROLLER NODE
# =========================
def mode_controller(state: SummarizerState):
    query = state.get("user_question", "")
    mode_str = classify_mode(query)
    print(f"Classified summary intent as: {mode_str}")

    try:
        mode_enum = SummaryMode(mode_str)  # convert string -> enum
    except ValueError:
        mode_enum = SummaryMode.OVERVIEW  # fallback default

    text = " ".join(state.get("retrieved_text_chunks", []))
    doc_tokens = max(1, len(text) // 4)

    config = MODE_CONFIG[mode_enum]
    budget = compute_budget(doc_tokens, config)

    # Ensure minimum budget for non-snapshot modes
    if mode_enum != SummaryMode.SNAPSHOT:
        budget = max(budget, 60)  # e.g., minimum 60 tokens

    detail = config["detail"]

    return {
        "summary_mode": mode_enum.value,  # store string for later use
        "token_budget": budget,
        "detail_level": detail
    }

# =========================
# GRAPH MODULE
# =========================
class SummarizationModule:
    def __init__(self, retriever, model):
        self.retriever = retriever
        self.model = model

        g = StateGraph(SummarizerState)
 
        # ---------- Nodes ----------
        g.add_node("mode_control", mode_controller)
        g.add_node("text_micro", lambda s: text_micro_agent(s, self.model))
        g.add_node("image_micro", lambda s: image_micro_agent(s, self.model))
        g.add_node("text_merge", lambda s: text_modality_agent(s, self.model))
        g.add_node("image_merge", lambda s: image_modality_agent(s, self.model))
        g.add_node("critical", lambda s: critical_agent(s, self.model))
        g.add_node("final", lambda s: summarization_agent(s, self.model))

        # ---------- Edges ----------
        g.add_edge(START, "mode_control")
        g.add_edge("mode_control", "text_micro")
        g.add_edge("mode_control", "image_micro")
        g.add_edge("text_micro", "text_merge")
        g.add_edge("image_micro", "image_merge")
        g.add_edge("text_merge", "critical")
        g.add_edge("image_merge", "critical")
        g.add_edge("critical", "final")
        g.add_edge("final", END)

        self.app = g.compile()

    def invoke(self, question: str, document: str = "all"):
        # Intercept generic queries to pull better context from the vector store
        search_query = question
        if len(question.split()) <= 5 and any(w in question.lower() for w in ["summary", "brief", "detail"]):
            search_query = "abstract introduction main contribution methodology conclusion"

        # Use the intercepted query for retrieval
        retrieved = self.retriever.query(search_query, k_text=6, k_image=4, document=document)

        state = {
            "llm_calls": 0,
            "intent": "summary",
            "user_question": question,  # Keep the original question for the LLM
            "retrieved_text_chunks": retrieved.get("text", []),
            "retrieved_images": retrieved.get("images", []),
            "image_captions": retrieved.get("captions", []),  # Extract captions if your retriever provides them
            "text_chunk_summaries": [],                       # Initialize empty list for text summaries
            "image_answers": [],                              # Initialize empty list for image answers
            "text_summary": "",
            "image_summary": "",
            "cross_modal_analysis": {},
            "final_summary": {},
            "general_context": "",
        }

        result = self.app.invoke(state)
        return result["final_summary"]