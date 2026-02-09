from langchain.messages import AnyMessage, HumanMessage, AIMessage, SystemMessage
import json 

import python.src.config.qa_prompts as qa_prompts
from src.config.model import model
from src.utils.json import clean_json_string

def text_agent(state:dict):
    critical_agent_output = json.loads(clean_json_string(state.get("ca_output", "")))
    critical_text_info = critical_agent_output.get("text")
    agent_answer:AIMessage = model.invoke(
        [
            SystemMessage(
                content=qa_prompts.TA_SYSTEM_PROMPT
            ),
            HumanMessage(
                content=f"""
                    Question: {state.get("user_question", "")}
                    Critical Text Information: {critical_text_info}
                    Textual Content: {state.get("retrieved_text", "No text provided")}
                """
            )
        ]
    )
    print("TEXT AGENT ANSWER: ", agent_answer)
    return {
        "messages": [agent_answer],
        "llm_calls": 1,
        "ta_output": agent_answer.content
    }