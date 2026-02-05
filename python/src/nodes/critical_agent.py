from langchain.messages import AnyMessage, HumanMessage, AIMessage, SystemMessage

import src.config.prompts as prompts
from src.config.model import model

def critical_agent(state:dict):
    agent_answer:AIMessage = model.invoke(
        [
            SystemMessage(
                content=prompts.CA_SYSTEM_PROMPT
            ),
            HumanMessage(
                content=f"""
                    Question: {state.get("user_question", "")}
                    Preliminary Answer: {state.get("ga_output"), ""}
                    Textual Content: {state.get("retrieved_text", "No text provided")}
                """
            )
        ]
    )
    
    return {
        "messages": [agent_answer],
        "llm_calls": 1,
        "ca_output": agent_answer.content
    }
    
    