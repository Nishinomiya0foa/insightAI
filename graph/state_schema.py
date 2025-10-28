# core/state_schema.py
from typing import TypedDict, List, Optional, Dict

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
