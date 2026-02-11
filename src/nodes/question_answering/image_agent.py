from langchain.messages import SystemMessage, HumanMessage, AIMessage
from src.config.model import model
import src.config.summarization_prompts as prompts
import src.config.qa_prompts as qprompts
import json
from src.utils.json import clean_json_string


def image_agent(state: dict, model=model):
    intent = state.get("intent", "qa")
    system_prompt = prompts.IA_SYSTEM_PROMPT if intent == "summary" else qprompts.QA_GA_SYSTEM_PROMPT

    critical_output = json.loads(clean_json_string(state.get("ca_output", "{}")))
    critical_image_info = critical_output.get("image", "")

    msg_content = [
        {"type": "text", "text": f"Question: {state.get('user_question', '')}\nCritical Image Info: {critical_image_info}"}
    ]

    # Attach base64 images if available
    for img_b64 in state.get("retrieved_images", []):
        msg_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}})

    agent_answer: AIMessage = model.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(msg_content)
    ])

    key = "image_summary" if intent == "summary" else "image_answer"
    return {
        "messages": [agent_answer],
        "llm_calls": 1,
        key: agent_answer.content
    }