from langchain.messages import AnyMessage, HumanMessage, AIMessage, SystemMessage

import python.src.config.qa_prompts as qa_prompts
from src.config.model import model

def critical_agent(state:dict):
    
    msg_content = [
        {
            "type": "text", 
            "text": f"""
                Question: {state.get("user_question", "")}
                Preliminary Answer: {state.get("ga_output"), ""}
                Textual Content: {state.get("retrieved_text", "No text provided")}
            """
        }
    ]
    if state.get("retrieved_images"):
        for img_b64 in state.get("retrieved_images"):
            msg_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_b64}"}
            })
    agent_answer:AIMessage = model.invoke(
        [
            SystemMessage(
                content=qa_prompts.CA_SYSTEM_PROMPT
            ),
            HumanMessage(
                msg_content
            )
        ]
    )
    print("CRITICAL AGENT ANSWER: ", agent_answer)
    return {
        "messages": [agent_answer],
        "llm_calls": 1,
        "ca_output": agent_answer.content
    }
    
    