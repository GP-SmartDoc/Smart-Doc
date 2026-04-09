from langgraph.graph import StateGraph, START, END
from langchain.messages import HumanMessage

from src.nodes.visualization.generating_agent import generating_agent
from src.nodes.visualization.regenerating_agent import regenerating_agent
from src.nodes.visualization.revising_agent import revising_agent
from src.nodes.visualization.description_agent import description_agent

from src.states.visualization_state import DiagramType
from src.states.visualization_RAG_state import VisualizationGraphState
from src.utils.strings import remove_thinking_from_content

def apply_dark_theme(diagram: str) -> str:
    """
    Safely apply dark styling to any Mermaid diagram
    without causing syntax errors.
    Works with:
    flowchart
    classDiagram
    sequenceDiagram
    stateDiagram
    erDiagram
    mindmap
    pie
    """

    if not diagram:
        return diagram

    # normalize
    d = diagram.strip()

    # detect diagram type
    first_line = d.split("\n", 1)[0].lower()

    # universal safe theme (works in mermaid v10.9.5)
    theme_block = """
        %%{init: {
        "theme": "dark",
        "themeVariables": {
        "primaryColor": "#1f2937",
        "primaryTextColor": "#ffffff",
        "primaryBorderColor": "#ffffff",
        "lineColor": "#ffffff",
        "textColor": "#ffffff"
        }}}%%
        """

    # avoid adding twice
    if "%%{init:" in d:
        return d

    return theme_block + "\n" + d
# -----------------------------
# FLOW CONTROL
# -----------------------------

def should_continue(state: VisualizationGraphState):

    if state["done"]:
        return END

    return "regenerator"


# -----------------------------
# GRAPH DEFINITION
# -----------------------------

builder = StateGraph(VisualizationGraphState)

builder.add_node("describe", description_agent)
builder.add_node("generator", generating_agent)
builder.add_node("revisor", revising_agent)
builder.add_node("regenerator", regenerating_agent)


# flow

builder.add_edge(START, "describe")

builder.add_edge("describe", "generator")

builder.add_edge("generator", "revisor")

builder.add_conditional_edges(
    "revisor",
    should_continue
)

builder.add_edge("regenerator", "revisor")


visualization_graph = builder.compile()


# -----------------------------
# MODULE
# -----------------------------

class VisualizationModule:

    def __init__(self, retriever):

        self.retriever = retriever
        self.graph = visualization_graph


    def invoke(
        self,
        request: str,
        diagram_type: DiagramType,
        document: str = "all"
    ) -> str:


        # -----------------------------
        # RAG Retrieval
        # -----------------------------

        retrieved = self.retriever.query(
            request,
            k_text=6,
            k_image=4,
            document=document
        )


        # -----------------------------
        # INITIAL STATE
        # -----------------------------

        initial_state: VisualizationGraphState = {

            "messages": [HumanMessage(content=request)],

            "llm_calls": 0,

            "user_request": request,

            "diagram_type": diagram_type,

            "retrieved_chunks": retrieved.get("text", []),

            "description": "",

            "generator_output": "",

            "revisor_output": "",

            "regenerator_output": "",

            "done": False
        }


        # -----------------------------
        # RUN GRAPH
        # -----------------------------

        final_state = self.graph.invoke(initial_state)


        # -----------------------------
        # RETURN FINAL DIAGRAM
        # -----------------------------

        if final_state.get("regenerator_output"):

            return remove_thinking_from_content(
                final_state["regenerator_output"]
            )

        diagram = remove_thinking_from_content(
        final_state.get("generator_output")
        )

        diagram = apply_dark_theme(diagram)

        return diagram


__all__ = ["VisualizationModule"]