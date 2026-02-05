from langchain.messages import AnyMessage, HumanMessage, AIMessage, SystemMessage
import base64
import json

import src.config.prompts as prompts
from src.config.model import model
from src.utils.json import clean_json_string

def image_agent(state:dict):
    # The critical agents returns a json
    critical_agent_output = json.loads(clean_json_string(state.get("ca_output", "")))
    critical_image_info = critical_agent_output.get("image")
    
    msg_content = [
        {"type": "text", "text": f"Textual Content: {state.get('retrieved_text')}\nQuestion: {state.get('user_question')}"}
    ]
    if state.get("retrieved_images"):
        for img_b64 in state.get("retrieved_images"):
            msg_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_b64}"}
            })
            
    agent_answer:AIMessage = model.invoke(
        [
            SystemMessage(
                content=prompts.IA_SYSTEM_PROMPT
            ),
            HumanMessage(
                content=f"""
                    Question: {state.get("user_question", "")}
                    Critical Image Information: {critical_image_info}
                    Image Content: {state.get("retrieved_images", "No images provided")}
                """
            )
        ]
    )

    return {
        "messages": [agent_answer],
        "llm_calls": 1,
        "ia_output": agent_answer.content
    } 