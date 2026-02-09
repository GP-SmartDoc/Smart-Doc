from typing_extensions import TypedDict, Annotated
import operator

class SlideGenerationGraphState(TypedDict):
    messages:Annotated[list[str], operator.add]
    llm_calls:Annotated[int, operator.add]

    #user_question:str
    retrieved_text:str
    retrieved_images:list[str] # <-- ?????

    Text_Summarizer_output:str
    Image_Captioner_output:str
    Code_Generator_output:str # without review

    # --- ADD THIS FIELD ---
    json_presentation_data: str # Stores the structured JSON for the PPTX renderer
    # ----------------------
    
    Code_Reviewer_output:str
    Page_Reviewer_output:str
    Code_Generator_output_Reviewed:str