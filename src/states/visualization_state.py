from typing_extensions import TypedDict, Annotated
import operator
from enum import Enum

class DiagramType(str, Enum):
    FLOWCHART = "flowchart"
    SEQUENCE = "sequence"
    STATE = "state"
    CLASS = "class"
    ER = "er"
    
class VisualizationGraphState(TypedDict):
    messages: Annotated[list[str], operator.add]
    llm_calls: Annotated[int, operator.add]

    description:str
    diagram_type:DiagramType
    
    generator_output:str
    revisor_output:str
    regenerator_output:str
    
    done:bool
