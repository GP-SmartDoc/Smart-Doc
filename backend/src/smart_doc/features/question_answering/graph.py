from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict, Annotated
import operator
from functools import partial

from smart_doc.features.question_answering.agents.general_agent import general_agent
from smart_doc.features.question_answering.agents.text_agent import text_agent
from smart_doc.features.question_answering.agents.image_agent import image_agent
from smart_doc.features.question_answering.agents.critical_agent import critical_agent
from smart_doc.features.question_answering.agents.qa_agent import qa_agent
from smart_doc.features.question_answering.agents.qa_complexity_agent import qa_complexity_evaluator_agent

class QAState(TypedDict):
    llm_calls: Annotated[int, operator.add]
    intent: str
    user_question: str
    document: str
    retrieved_text_chunks: list
    retrieved_images: list
    text_answer: str
    image_answer: str
    cross_modal_analysis: dict
    final_answer: dict

class QuestionAnsweringModule:
    def __init__(self, retriever):
        self.retriever = retriever
        
        # QA StateGraph
        g = StateGraph(QAState)

        # QA flow nodes
        g.add_node("general", general_agent)
        g.add_node("text", text_agent)
        g.add_node("image", image_agent)
        g.add_node("critical", critical_agent)
        g.add_node("final", qa_agent)
        
        # Add the complexity node with the retriever passed in
        g.add_node("complexity", partial(qa_complexity_evaluator_agent, retriever=self.retriever))

        # 🔧 FIX: Linearized the execution edges to prevent overlapping steps
        g.add_edge(START, "general")
        g.add_edge("general", "text")
        g.add_edge("text", "image")  
        g.add_edge("image", "critical")
        g.add_edge("critical", "final")
        g.add_edge("final", "complexity")
        g.add_edge("complexity", END)

        self.app = g.compile()
        
    def invoke(self, question: str, document: str = "all"):
        retrieved = self.retriever.query(question, k_text=6, k_image=4, document=document)

        state: QAState = {
            "llm_calls": 1,
            "intent": "qa",
            "user_question": question,
            "document": document, 
            "retrieved_text_chunks": retrieved.get("text", []),
            "retrieved_images": retrieved.get("images", []),
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