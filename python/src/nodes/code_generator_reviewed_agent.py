from langchain.messages import AnyMessage, HumanMessage, AIMessage, SystemMessage

import src.config.prompts as prompts
from src.config.model import model

def Code_Generator_Reviewed(state:dict)   :
    agent_answer:AIMessage = model.invoke(
        [
            SystemMessage(
                    content=prompts.CGR_SYSTEM_PROMPT
                ),
            HumanMessage(
                    content=prompts.CGR_USER_PROMPT.replace(
                        "<TextSummary.md>",
                        state["Text_Summarizer_output"]
                    ).replace(
                        "<ImageCaption.md>",
                        state["Image_Captioner_output"]
                    ).replace(
                        "<CodeReview.md>",
                        state["Code_Reviewer_output"]
                    )
            )
        ]
    )
    return {
        "messages": [agent_answer],
        "llm_calls": 1,
        "Code_Generator_output_Reviewed": agent_answer.content
    }