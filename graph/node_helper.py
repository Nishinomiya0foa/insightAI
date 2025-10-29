from .memory_manager import get_recent_feedbacks

def make_prompt_from_feedback_memory(session_id):
    """结合历史反馈记录, 生成prompt"""
    recent_feedbacks = get_recent_feedbacks(session_id)
    satisfied_feedbacks = [f for f in recent_feedbacks if f.get("feedback") == "satisfied"]
    if satisfied_feedbacks:
        summary = "\n".join([
            f"用户喜欢这类回答: {f.get('answer')[:120]}..." for f in satisfied_feedbacks
        ])
        prompt = f"\n用户过去对以下回答表示满意, 在生成回答时参考其风格与详细程度:\n{summary}\n"
        return prompt
    return ""
