from langchain.messages import AnyMessage, HumanMessage, AIMessage, SystemMessage

import src.config.prompts as prompts
from src.config.model import model

def general_agent(state:dict):
    agent_answer:AIMessage = model.invoke(
        [
            SystemMessage(
                content= prompts.GA_SYSTEM_PROMPT
            ),
            HumanMessage(
                content=f"""
                    Textual Content: {state.get("retrieved_text", "No text provided")}
                    Question: {state.get("user_question", "")}
                """
            )
        ]
    )
    
    return {
        "messages": [agent_answer],
        "llm_calls": 1,
        "ga_output": agent_answer.content
    }