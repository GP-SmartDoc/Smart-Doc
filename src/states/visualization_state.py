from typing_extensions import TypedDict, Annotated
import operator

class VisualizationGraphState(TypedDict):
    messages: Annotated[list[str], operator.add]
    llm_calls: Annotated[int, operator.add]

    description:str

    generator_output:str
    revisor_output:str
    regenerator_output:str
    
    done:bool
