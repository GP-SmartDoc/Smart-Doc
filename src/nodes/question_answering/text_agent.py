from langchain.messages import SystemMessage, HumanMessage, AIMessage
from src.config.model import model
import src.config.summarization_prompts as prompts
import src.config.qa_prompts as qprompts
import json
from src.utils.strings import clean_json_string


def text_agent(state: dict, model=model):
    intent = state.get("intent", "qa")
    system_prompt = prompts.TA_SYSTEM_PROMPT if intent == "summary" else qprompts.QA_TA_SYSTEM_PROMPT

    critical_output = json.loads(clean_json_string(state.get("ca_output", "{}")))
    critical_text = critical_output.get("text", "")

    content = f"""
        Question: {state.get('user_question', '')}
        Critical Text Info: {critical_text}
        Textual Content: {state.get('retrieved_text_chunks', [])}
    """

    agent_answer: AIMessage = model.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=content)
    ])

    key = "text_summary" if intent == "summary" else "text_answer"
    return {
        "messages": [agent_answer],
        "llm_calls": 1,
        key: agent_answer.content
    }