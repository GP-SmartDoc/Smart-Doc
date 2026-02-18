from langchain.messages import SystemMessage, HumanMessage
from config.model import model
import config.summarization_prompts as prompts
import config.qa_prompts as qprompts
import json
from utils.strings import clean_json_string


def image_micro_agent(state, model):
    captions = state.get("image_captions", [])

    # HARD STOP: do not hallucinate if no captions
    if not captions:
        return {
            "image_answers": [],
            "llm_calls": 0
        }

    merged_caption = "\n".join(
        f"- Region {i+1}: {cap}"
        for i, cap in enumerate(captions)
    )

    response = model.invoke([
        SystemMessage(content=
            "You analyze technical diagrams and architectures. "
            "Infer stages, components, workflows, and relationships "
            "ONLY from the provided descriptions. "
            "Do NOT say you cannot see the image."
        ),
        HumanMessage(content=f"""
User Question:
{state.get('user_question','')}

Visual Descriptions:
{merged_caption}

Infer the architecture or process clearly.
""")
    ])

    return {
        "image_answers": [response.content],
        "llm_calls": 1
    }


def image_modality_agent(state: dict, model):
    intent = state.get("intent", "qa")

    # ❌ REMOVE image_summaries from QA path completely
    image_answers = state.get("image_answers", [])

    # If image_micro_agent produced nothing, skip cleanly
    if not image_answers:
        return {
            "image_answer": "",
            "llm_calls": 0
        }

    # Merge micro reasoning into single evidence block
    image_text = "\n".join(image_answers)

    if intent == "summary":
        system_prompt = prompts.IA_MODALITY_SYSTEM_PROMPT
        payload = {
            "question": state.get("user_question", ""),
            "image_evidence": image_text
        }
        out_key = "image_summary"
    else:
        system_prompt = qprompts.QA_GA_SYSTEM_PROMPT
        payload = {
            "question": state.get("user_question", ""),
            "image_evidence": image_text
        }
        out_key = "image_answer"

    resp = model.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=json.dumps(payload))
    ])

    return {
        out_key: resp.content,
        "llm_calls": 1
    }