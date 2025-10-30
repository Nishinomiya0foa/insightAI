from pydantic import BaseModel


class FeedbackRequest(BaseModel):
    session_id: str
    satisfied: bool
    feedback: str | None = None
