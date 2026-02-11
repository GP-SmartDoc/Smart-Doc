from langchain.messages import AnyMessage, HumanMessage, AIMessage, SystemMessage

import src.config.visualization_prompts as prompts
from src.config.model import model

def Code_Generator(state: dict):
    image_list = "\n".join(
        f"{i+1}. {img}" for i, img in enumerate(state["retrieved_images"])
    )

    user_prompt = (
        prompts.CG_USER_PROMPT
        .replace("<TextSummary.md>", state["Text_Summarizer_output"])
        .replace("<ImageCaption.md>", state["Image_Captioner_output"])
        .replace("<ImageList>", image_list)
    )

    agent_answer: AIMessage = model.invoke([
        SystemMessage(content=prompts.CG_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt)
    ])

    return {
        "messages": [agent_answer],
        "llm_calls": 1,
        "Code_Generator_output": agent_answer.content
    }