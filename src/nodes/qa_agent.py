from langchain.messages import SystemMessage, HumanMessage, AIMessage
import json
import src.config.qa_prompts as qa_prompts
import src.config.summarization_prompts as summary_prompts

def qa_agent(state: dict, model) -> dict:
    intent = state.get("intent", "summary")

    if intent == "qa":
        text_info = state.get("ta_output", state.get("text_answer", ""))
        image_info = state.get("ia_output", state.get("image_answer", ""))
        general_context = state.get("general_context", "")

        agent_answer: AIMessage = model.invoke([
            SystemMessage(content=qa_prompts.QA_FINAL_SYSTEM_PROMPT),
            HumanMessage(
                content=f"""
Question: {state.get("user_question", "")}
Text Info: {text_info}
Image Info: {image_info}
General Context: {general_context}
"""
            )
        ])

        return {
            "messages": [agent_answer],
            "llm_calls": 1,
            "qa_output": agent_answer.content
        }

    else:
        system_prompt = summary_prompts.SA_SYSTEM_PROMPT
        resp = model.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=json.dumps({
                "question": state.get("user_question", ""),
                "text_summary": state.get("text_chunk_summaries", []),
                "image_summary": state.get("image_summaries", []),
                "general_context": state.get("general_context", "")
            }))
        ])
        return {
            "final_answer": resp.content,
            "llm_calls": 1
        }
