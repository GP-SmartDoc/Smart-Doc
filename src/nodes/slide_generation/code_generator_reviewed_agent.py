from langchain.messages import AnyMessage, HumanMessage, AIMessage, SystemMessage

import src.config.slide_generation_prompts as prompts
from src.config.model import model

def Code_Generator_Reviewed(state:dict):
    
    # Safely get the outputs using .get() to avoid KeyError crashes
    text_summary = state.get("Text_Summarizer_output", "")
    image_caption = state.get("Image_Captioner_output", "")
    code_review = state.get("Code_Reviewer_output", "")
    
    # CRITICAL FIX: Grab the JSON output from the first agent
    previous_json = state.get("Code_Generator_output", "")
    
    if not previous_json:
        print("WARNING: Code_Generator_output is empty in the state dictionary!")

    agent_answer:AIMessage = model.invoke(
        [
            SystemMessage(
                    content=prompts.CGR_SYSTEM_PROMPT
                ),
            HumanMessage(
                    content=prompts.CGR_USER_PROMPT.replace(
                        "<TextSummary.md>", state["Text_Summarizer_output"]
                    ).replace(
                        "<ImageCaption.md>", state["Image_Captioner_output"]
                    ).replace(
                        "<CodeReview.md>", state["Code_Reviewer_output"]
                    ).replace(
                        "<PreviousJSON>", state.get("Code_Generator_output", "") # ADD THIS LINE
                    )
            )
        ]
    )
    
    return {
        "messages": [agent_answer],
        "llm_calls": 1,
        "Code_Generator_output_Reviewed": agent_answer.content
    }