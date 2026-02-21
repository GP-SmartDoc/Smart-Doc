from langchain.messages import SystemMessage, HumanMessage
from src.config.model import model
import src.config.summarization_prompts as prompts
import src.config.qa_prompts as qprompts
import json

def critical_agent(state: dict, model):
    if state.get("cross_modal_analysis"):
        return {"llm_calls": 0}

    intent = state.get("intent", "qa")
    text_data = state.get("text_summary", "")
    image_data = state.get("image_summary", "")
    
    # ---------------------------
    # FIX 1: Explicitly handle empty data 
    # Prevents the LLM from hallucinating missing image context
    # ---------------------------
    if not image_data.strip():
        image_data = "No image evidence provided."
        
    if not text_data.strip():
        text_data = "No text evidence provided."

    payload = {
        "question": state.get("user_question", ""),
        "text": text_data,   
        "image": image_data,
        "detail_level": state.get("detail_level", 2)  
    }

    # ---------------------------
    # FIX 2: Dynamic prompt routing based on intent
    # ---------------------------
    if intent == "summary":
        system_prompt = prompts.CA_SYSTEM_PROMPT
    else:
        # Fallback to QA prompt if intent is QA
        system_prompt = getattr(qprompts, "QA_CA_SYSTEM_PROMPT", prompts.CA_SYSTEM_PROMPT)

    resp = model.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=json.dumps(payload))
    ])

    # ---------------------------
    # FIX 3: Safe JSON Parsing
    # Strips markdown blocks before parsing
    # ---------------------------
    try:
        content = resp.content.strip()
        if content.startswith("```json"):
            content = content[7:-3].strip()
        elif content.startswith("```"):
            content = content[3:-3].strip()
            
        analysis = json.loads(content)
    except Exception as e:
        print(f"[DEBUG] JSON Parse Error in Critical Agent: {e}")
        # Fallback to the raw inputs if parsing completely fails
        analysis = {
            "text": text_data, 
            "image": image_data
        }

    print("[DEBUG] Critical analysis:", analysis)

    return {
        "cross_modal_analysis": analysis,
        "llm_calls": 1
    }