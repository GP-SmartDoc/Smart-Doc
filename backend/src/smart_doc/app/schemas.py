from pydantic import BaseModel


class ChatRequest(BaseModel):
    message: str
    document: str
    mode: str
    summary_mode: str = "overview"
    user_id: str | None = None

