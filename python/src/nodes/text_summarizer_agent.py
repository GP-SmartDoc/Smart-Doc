from langchain.messages import AnyMessage, HumanMessage, AIMessage, SystemMessage

import src.config.prompts as prompts
from src.config.model import model

def Text_Summarizer(state:dict)   :
    agent_answer:AIMessage = model.invoke(
        [
            SystemMessage(
                    content=prompts.TS_SYSTEM_PROMPT
                ),
            HumanMessage(
                    content=prompts.TS_USER_PROMPT.replace(
                        "<Document text>",
                        state["retrieved_text"]
                    )
                )
        ]
    )
    return {
        "messages": [agent_answer],
        "llm_calls": 1,
        "Text_Summarizer_output": agent_answer.content
    }