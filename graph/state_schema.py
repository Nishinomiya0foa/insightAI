from typing import TypedDict, List, Optional, Dict
from pydantic import BaseModel, Field


class InsightState(TypedDict, total=False):
    session_id: str
    question: Optional[str]
    documents: List[str]
    retrieved_docs: List[str]
    context: str
    answer: str
    user_intent: str
    suggestion: str
    memory_hits: List[Dict]
    new_memory_entry: Dict
    feedback: Optional[str]
    satisfied: bool
    feedbacks: List[str]


class IntentResult(BaseModel):
    intents: List[str] = Field(..., description="推测的用户的下一步可能的提问")