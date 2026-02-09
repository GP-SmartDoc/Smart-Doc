from typing_extensions import TypedDict, Annotated
import operator

class SummarizerState(TypedDict):
    llm_calls: Annotated[int, operator.add]
    intent: str
    user_question: str
    retrieved_text_chunks: list
    retrieved_images: list
    text_chunk_summaries: list
    image_summaries: list
    text_summary: str
    image_summary: str
    cross_modal_analysis: dict
    final_summary: dict
