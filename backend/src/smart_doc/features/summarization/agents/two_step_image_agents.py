from langchain.messages import SystemMessage, HumanMessage
from smart_doc.core.models import image_model as model
import smart_doc.features.summarization.prompts as prompts
import json


def image_analyst_agent(state: dict, model=model):
    captions = state.get("image_captions", [])
    if not captions:
        return {"image_answers": [], "llm_calls": 0}

    detail_level = state.get("detail_level", 2)

    sentences_per_caption = {1: 1, 2: 2, 3: 4}
    max_sents = sentences_per_caption.get(detail_level, 2)

    merged_caption = "\n".join(
        f"- Region {i+1}: {cap}" for i, cap in enumerate(captions)
    )
    system_prompt = prompts.IA_SYSTEM_PROMPT
    resp = model.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"""
User Question:
{state.get('user_question', '')}

Image Descriptions:
{merged_caption}

Detail Level: {detail_level} (Max {max_sents} sentences per image)
""")
    ])

    return {
        "image_answers": [resp.content],
        "llm_calls": 1
    }


def image_aggregator_agent(state: dict, model=model):
    image_answers = state.get("image_answers", [])

    # Text-only documents should still flow through synthesis cleanly.
    if not image_answers:
        return {
            "image_summary": "",
            "llm_calls": 0
        }

    image_text = "\n".join(image_answers)
    payload = {
        "question": state.get("user_question", ""),
        "detail_level": state.get("detail_level", 2),
        "image_evidence": image_text
    }

    resp = model.invoke([
        SystemMessage(content=prompts.IA_MODALITY_SYSTEM_PROMPT),
        HumanMessage(content=json.dumps(payload))
    ])

    return {
        "image_summary": resp.content,
        "llm_calls": 1
    }
