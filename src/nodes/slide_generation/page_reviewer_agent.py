from langchain.messages import AnyMessage, HumanMessage, AIMessage, SystemMessage

import config.slide_generation_prompts as prompts
from src.config.model import model

def Page_Reviewer(state:dict)   :
    agent_answer:AIMessage = model.invoke(
        [
            SystemMessage(
                    content=prompts.PR_SYSTEM_PROMPT
                ),
            HumanMessage(
                    content=prompts.PR_USER_PROMPT.replace(
                        "<ImageCaption.md>",
                        state["Image_Captioner_output"]
                    ).replace(
                        "<Document Images>",
                        "\n".join(state["retrieved_images"]))
                    )
        ]
    )
    
    return {
        "messages": [agent_answer],
        "llm_calls": 1,
        "Page_Reviewer_output": agent_answer.content
    }