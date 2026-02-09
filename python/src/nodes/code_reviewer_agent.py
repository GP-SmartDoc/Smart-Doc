from langchain.messages import AnyMessage, HumanMessage, AIMessage, SystemMessage

import src.config.prompts as prompts
from src.config.model import model

def Code_Reviewer(state:dict)   :
    agent_answer:AIMessage = model.invoke(
        [
            SystemMessage(
                    content=prompts.CR_SYSTEM_PROMPT
                ),
            HumanMessage(
                    content=prompts.CR_USER_PROMPT.replace(
                        "<TextSummary.md>",
                        state["Text_Summarizer_output"]
                    ).replace(
                        "<ImageCaption.md>",
                        state["Image_Captioner_output"]
                    ).replace(
                        "<SlidevCode.md>",
                        state["Code_Generator_output"]
                    )
            )
        ]
    )
    return {
        "messages": [agent_answer],
        "llm_calls": 1,
        "Code_Reviewer_output": agent_answer.content
    }