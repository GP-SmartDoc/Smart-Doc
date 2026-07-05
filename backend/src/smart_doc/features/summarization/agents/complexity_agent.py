import json
from langchain.messages import SystemMessage, HumanMessage
from smart_doc.core.models import text_model as model
from smart_doc.features.summarization.prompts import COMPLEXITY_EVALUATOR_SYSTEM_PROMPT
from smart_doc.utils.helper import safe_json_parse
from smart_doc.features.visualization.rag_graph import VisualizationModule

def complexity_evaluator_agent(state: dict, retriever=None):
    """
    Evaluates text complexity and dynamically calls the visualization module 
    if a diagram would significantly enhance clarity.
    """
    final_summary_dict = state.get("final_summary", {})
    summary_text = final_summary_dict.get("Answer", "")
    
    if not summary_text or state.get("summary_mode") == "snapshot":
        # Snapshot mode is too brief to require diagrams
        return {"final_summary": final_summary_dict, "llm_calls": 0}

    # Evaluate complexity
    resp = model.invoke([
        SystemMessage(content=COMPLEXITY_EVALUATOR_SYSTEM_PROMPT),
        HumanMessage(content=f"Analyze this summary:\n\n{summary_text}")
    ])
    
    evaluation = safe_json_parse(resp.content, {"needs_diagram": False})
    
    # Keeping the forced True condition as you requested for testing
    if evaluation.get("needs_diagram") and retriever is not None:
        try:
            # Instantiate the visualization module using the shared retriever
            viz_module = VisualizationModule(retriever=retriever)
            
            # Extract requested parameters
            diagram_type = evaluation.get("diagram_type", "flowchart")
            viz_request = evaluation.get("visualization_request", state.get("user_question"))
            
            # Run the visual diagram generator graph
            mermaid_code = viz_module.invoke(
                request=viz_request,
                diagram_type=diagram_type,
                document="all"
            )
            
            # 🔧 FIX 1: Strip existing markdown formatting from the LLM output
            if "```mermaid" in mermaid_code:
                mermaid_code = mermaid_code.split("```mermaid")[1].split("```")[0].strip()
            elif "```" in mermaid_code:
                mermaid_code = mermaid_code.replace("```", "").strip()
            
            # Inject diagram content directly into the structured final summary dictionary
            final_summary_dict["Diagram"] = mermaid_code
            final_summary_dict["DiagramType"] = diagram_type
            final_summary_dict["DiagramReasoning"] = evaluation.get("diagram_reasoning")

            # 🔧 FIX 2: Removed the block that forcibly appends the diagram to final_summary_dict["Answer"].
            # routes.py's build_summary_reply() will now cleanly handle formatting it!
            
        except Exception as e:
            print(f"Error during autonomous diagram generation: {e}")
            
    return {
        "final_summary": final_summary_dict,
        "llm_calls": 1
    }