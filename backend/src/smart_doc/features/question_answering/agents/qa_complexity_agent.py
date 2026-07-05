import json
from langchain.messages import SystemMessage, HumanMessage
from smart_doc.core.models import text_model as model
from smart_doc.features.summarization.prompts import COMPLEXITY_EVALUATOR_SYSTEM_PROMPT
from smart_doc.utils.helper import safe_json_parse
from smart_doc.features.visualization.rag_graph import VisualizationModule

def qa_complexity_evaluator_agent(state: dict, retriever=None):
    """
    Evaluates QA complexity and dynamically calls the visualization module 
    if a diagram would significantly enhance clarity of the answer.
    """
    final_answer_data = state.get("final_answer", {})
    
    # Handle the fact that qa_agent returns a JSON string
    if isinstance(final_answer_data, str):
        final_answer_data = safe_json_parse(final_answer_data, {"Answer": final_answer_data})
        
    answer_text = final_answer_data.get("Answer", "")
    
    if not answer_text:
        return {"final_answer": final_answer_data, "llm_calls": 0}

    # Evaluate complexity using the existing summarization prompt
    resp = model.invoke([
        SystemMessage(content=COMPLEXITY_EVALUATOR_SYSTEM_PROMPT),
        HumanMessage(content=f"Analyze this answer:\n\n{answer_text}")
    ])
    
    evaluation = safe_json_parse(resp.content, {"needs_diagram": False})
    
    # If complex, generate the diagram
    if evaluation.get("needs_diagram") and retriever is not None:
        try:
            viz_module = VisualizationModule(retriever=retriever)
            diagram_type = evaluation.get("diagram_type", "flowchart")
            viz_request = evaluation.get("visualization_request", state.get("user_question"))
            
            mermaid_code = viz_module.invoke(
                request=viz_request,
                diagram_type=diagram_type,
                document="all"
            )
            
            # Strip markdown artifacts
            if "```mermaid" in mermaid_code:
                mermaid_code = mermaid_code.split("```mermaid")[1].split("```")[0].strip()
            elif "```" in mermaid_code:
                mermaid_code = mermaid_code.replace("```", "").strip()
            
            # Inject diagram directly into the text so routes.py's extract_answer() catches it
            final_answer_data["Answer"] = (
                final_answer_data["Answer"].strip()
                + "\n\n```mermaid\n"
                + mermaid_code
                + "\n```"
            )
            
        except Exception as e:
            print(f"Error during QA diagram generation: {e}")
            
    return {
        "final_answer": final_answer_data,
        "llm_calls": 1
    }