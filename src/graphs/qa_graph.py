from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict, Annotated
import operator

from src.nodes.general_agent import general_agent
from src.nodes.text_agent import text_micro_agent, text_modality_agent
from src.nodes.image_agent import image_micro_agent, image_modality_agent
from src.nodes.critical_agent import critical_agent
from src.nodes.qa_agent import qa_agent
from src.graphs.summary_graph import SummarizationModule

# ----------------------------
# State Definition
# ----------------------------
class QAState(TypedDict):
    llm_calls: Annotated[int, operator.add]
    intent: str
    user_question: str

    retrieved_text_chunks: list
    retrieved_images: list
    image_captions: list  

    text_answers: list
    image_answers: list

    text_answer: str
    image_answer: str

    cross_modal_analysis: dict
    final_answer: dict

# ----------------------------
# QA Module
# ----------------------------
class QuestionAnsweringModule:
    def __init__(self, retriever, model):
        self.retriever = retriever
        self.model = model

        # Initialize summarization module for pre-processing
        self.summarizer = SummarizationModule(retriever=retriever, model=model)

        # Build the QA StateGraph
        g = StateGraph(QAState)

        # -------- Nodes --------
        g.add_node("general", lambda s: general_agent(s, self.model))

        g.add_node("text_micro", lambda s: text_micro_agent(s, self.model))
        g.add_node("image_micro", lambda s: image_micro_agent(s, self.model))

        g.add_node("text_merge", lambda s: text_modality_agent(s, self.model))
        g.add_node("image_merge", lambda s: image_modality_agent(s, self.model))

        g.add_node("critical", lambda s: critical_agent(s, self.model))
        g.add_node("final", lambda s: qa_agent(s, self.model))

        # -------- Edges (fixed flow) --------
        g.add_edge(START, "general")

        g.add_edge("general", "text_micro")
        g.add_edge("general", "image_micro")

        g.add_edge("text_micro", "text_merge")
        g.add_edge("image_micro", "image_merge")

        # Ensure critical sees both modalities
        g.add_edge("text_merge", "image_merge")
        g.add_edge("image_merge", "critical")

        g.add_edge("critical", "final")
        g.add_edge("final", END)

        self.app = g.compile()

    # ----------------------------
    # Invoke
    # ----------------------------
    def invoke(self, question: str):
        # ---------------- Retrieve raw data ----------------
        retrieved = self.retriever.query(question, k_text=6, k_image=4)

        # ---------------- Initialize state ----------------
        state: QAState = {
            "llm_calls": 0,
            "intent": "qa",
            "user_question": question,

            "retrieved_text_chunks": retrieved.get("text", []),
            "retrieved_images": retrieved.get("images", []),
            "image_captions": retrieved.get("image_captions", []),

            "text_answers": [],
            "image_answers": [],

            "text_answer": "",
            "image_answer": "",

            "cross_modal_analysis": {},
            "final_answer": {},
        }

        # ---------------- Summarize text ----------------
        text_summary = self.summarizer.app.invoke({
            "llm_calls": 0,
            "intent": "summary",
            "user_question": question,
            "retrieved_text_chunks": state["retrieved_text_chunks"],
            "retrieved_images": [],  # image summaries are handled later
            "text_chunk_summaries": [],
            "image_summaries": [],
            "text_summary": "",
            "image_summary": "",
            "cross_modal_analysis": {},
            "final_summary": {}
        })["final_summary"]

        state["text_answers"].append(text_summary)

        # ---------------- Invoke main QA graph ----------------
        result = self.app.invoke(state)
        return result["final_answer"]
