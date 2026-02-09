from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict, Annotated
import operator
from src.nodes.general_agent import general_agent
from src.nodes.text_agent import text_micro_agent
from src.nodes.image_agent import image_micro_agent
from src.nodes.critical_agent import critical_agent
from src.nodes.qa_agent import qa_agent

class QAState(TypedDict):
    llm_calls: Annotated[int, operator.add]
    intent: str
    user_question: str
    retrieved_text_chunks: list
    retrieved_images: list
    text_answers: list
    image_answers: list
    cross_modal_analysis: dict
    final_answer: dict

class QuestionAnsweringModule:
    def __init__(self, retriever, model):
        self.retriever = retriever
        self.model = model

        g = StateGraph(QAState)

        g.add_node("general", lambda s: general_agent(s, self.model))
        g.add_node("text", lambda s: text_micro_agent(s, self.model))
        g.add_node("image", lambda s: image_micro_agent(s, self.model))
        g.add_node("critical", lambda s: critical_agent(s, self.model))
        g.add_node("final", lambda s: qa_agent(s, self.model))

        g.add_edge(START, "general")
        g.add_edge(START, "text")
        g.add_edge(START, "image")
        g.add_edge("general", "critical")
        g.add_edge("text", "critical")
        g.add_edge("image", "critical")
        g.add_edge("critical", "final")
        g.add_edge("final", END)

        self.app = g.compile()

    def invoke(self, question: str):
        retrieved = self.retriever.query(question, k_text=6, k_image=4)

        state: QAState = {
            "llm_calls": 0,
            "intent": "qa",
            "user_question": question,
            "retrieved_text_chunks": retrieved.get("text", []),
            "retrieved_images": retrieved.get("images", []),
            "text_answers": [],
            "image_answers": [],
            "cross_modal_analysis": {},
            "final_answer": {},
        }

        result = self.app.invoke(state)
        return result["final_answer"]
