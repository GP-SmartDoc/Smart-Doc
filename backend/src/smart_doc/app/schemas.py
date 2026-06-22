from pydantic import BaseModel


class ChatRequest(BaseModel):
    message: str
    document: str
    mode: str
    summary_mode: str = "overview"

