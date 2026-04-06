from typing_extensions import TypedDict, Annotated
import operator
from enum import Enum
from src.states.visualization_state import DiagramType

class VisualizationGraphState(TypedDict):

    messages: Annotated[list[str], operator.add]
    llm_calls: Annotated[int, operator.add]

    user_request:str   # NEW
    description:str
    diagram_type:DiagramType

    retrieved_chunks:list   # NEW

    generator_output:str
    revisor_output:str
    regenerator_output:str

    done:bool