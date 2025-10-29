from pydantic import BaseModel


class FeedbackRequest(BaseModel):
    session_id: str
    question: str
    answer: str
    satisfied: bool
    new_prompt: str | None = None
