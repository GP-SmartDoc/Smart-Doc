from langchain.messages import AnyMessage, HumanMessage, AIMessage, SystemMessage
import json 

import config.qa_prompts as qa_prompts
from config.model import model

def summarizing_agent(state:dict):
    agent_answer:AIMessage = model.invoke(
        [
            SystemMessage(
                content=qa_prompts.SA_SYSTEM_PROMPT
            ),
            HumanMessage(
                content=f"""
                    Question: {state.get("user_question", "")}
                    Preliminary Answer: {state.get("ga_output", "")}
                    Text Agent Answer: {state.get("ta_output", "")}
                """
            )
        ]
    )
    print("SUMMARIZING AGENT ANSWER: ", agent_answer)
    return {
        "messages": [agent_answer],
        "llm_calls": 1,
        "sa_output": agent_answer.content
    }