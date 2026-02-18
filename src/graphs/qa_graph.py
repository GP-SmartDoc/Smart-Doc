from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict, Annotated
import operator

from nodes.question_answering.general_agent import general_agent
from nodes.question_answering.text_agent import text_agent
from nodes.question_answering.image_agent import image_agent
from nodes.question_answering.critical_agent import critical_agent
from nodes.question_answering.qa_agent import qa_agent

class QAState(TypedDict):
    llm_calls: Annotated[int, operator.add]
    intent: str
    user_question: str
    retrieved_text_chunks: list
    retrieved_images: list
    image_captions: list
    text_answer: str
    image_answer: str
    cross_modal_analysis: dict
    final_answer: dict

class QuestionAnsweringModule:
    def __init__(self, retriever, model):
        self.retriever = retriever
        self.model = model

        # QA StateGraph
        g = StateGraph(QAState)

        # QA flow nodes
        g.add_node("general", lambda s: general_agent(s, self.model))
        g.add_node("text", lambda s: text_agent(s, self.model))
        g.add_node("image", lambda s: image_agent(s, self.model))
        g.add_node("critical", lambda s: critical_agent(s, self.model))
        g.add_node("final", lambda s: qa_agent(s, self.model))

        g.add_edge(START, "general")
        g.add_edge("general", "text")
        g.add_edge("general", "image")
        g.add_edge("text", "image")  
        g.add_edge("image", "critical")
        g.add_edge("critical", "final")
        g.add_edge("final", END)

        self.app = g.compile()

    def invoke(self, question: str, intent: str = "qa"):
        """
        intent: "qa" for full QA,
        """

        retrieved = self.retriever.query(question, k_text=6, k_image=4)

        state: QAState = {
            "llm_calls": 0,
            "intent": intent,
            "user_question": question,
            "retrieved_text_chunks": retrieved.get("text", []),
            "retrieved_images": retrieved.get("images", []),
            "image_captions": retrieved.get("image_captions", []),
            "text_answer": "",
            "image_answer": "",
            "cross_modal_analysis": {},
            "final_answer": {},
        }

        # ---------------- FULL QA MODE ----------------
        result = self.app.invoke(state)
        return {
            "Answer": result["final_answer"]  
        }
