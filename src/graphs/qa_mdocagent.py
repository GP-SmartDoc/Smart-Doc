from langgraph.graph import StateGraph, MessagesState, START, END
from langchain.messages import AnyMessage, HumanMessage, AIMessage

from nodes.question_answering_mdocagent.critical_agent import critical_agent
from nodes.question_answering_mdocagent.general_agent import general_agent
from nodes.question_answering_mdocagent.image_agent import image_agent
from nodes.question_answering_mdocagent.summarizing_agent import summarizing_agent
from nodes.question_answering_mdocagent.text_agent import text_agent
from states.QAState import QuestionAnsweringGraphState
from vector_store.chroma import rag

builder = StateGraph(QuestionAnsweringGraphState)

builder.add_node("general_agent", general_agent)
builder.add_node("critical_agent", critical_agent)
builder.add_node("text_agent", text_agent)
builder.add_node("image_agent", image_agent)
builder.add_node("summarizing_agent", summarizing_agent)

builder.add_edge(START, "general_agent")
builder.add_edge("general_agent", "critical_agent")
builder.add_edge("critical_agent", "text_agent")
builder.add_edge("critical_agent", "image_agent")
builder.add_edge("text_agent", "summarizing_agent")
builder.add_edge("image_agent", "summarizing_agent")
builder.add_edge("summarizing_agent", END)

qa_module = builder.compile()

def invoke_qa_workflow(prompt:str)->str:
    # Query RAG (Ensure RAG.py is also fixed to handle k_image=0 as per previous step)
    retrieved_data = rag.query(prompt, k_text=5, k_image=5)
    
    text_content = retrieved_data.get("text")
    retrieved_text = "\n".join(text_content)
    
    retrieved_images = retrieved_data.get("images")
    
    initial_state = {
        "messages": [HumanMessage(content=prompt)],
        "user_question": prompt,  
        "retrieved_text": retrieved_text, 
        "retrieved_images": retrieved_images,
        "llm_calls": 0
    }
    final_state = qa_module.invoke(initial_state)
    return final_state.get("sa_output", "")