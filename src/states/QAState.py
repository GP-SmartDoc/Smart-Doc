from typing_extensions import TypedDict, Annotated
import operator

class QuestionAnsweringGraphState(TypedDict):
    messages: Annotated[list[str], operator.add]
    llm_calls: Annotated[int, operator.add]

    user_question:str
    retrieved_text:str
    retrieved_images:list[str] 

    ga_output:str
    ca_output:str
    ta_output:str
    ia_output:str
    sa_output:str