from typing import TypedDict, List, Optional, Dict

class InsightState(TypedDict, total=False):
    question: Optional[str]
    documents: List[str]
    retrieved_docs: List[str]
    context: str
    answer: str
    user_intent: str
    suggestion: str
    memory: Dict
