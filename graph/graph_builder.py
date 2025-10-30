from typing import Dict

from langgraph.graph import StateGraph, START, END
from .state_schema import InsightState
from .nodes import (
    load_documents, build_vector_index, memory_read,
    retrieve_context, generate_answer, infer_intent,
    memory_write, record_satisfied, record_unsatisfied,
    feedback_read, summary_answer
)


def decide_where_start(state: Dict) -> str:
    """决定开始后的第一个node"""
    if state.get("satisfied") is True:
        return "record_satisfied"
    elif state.get("satisfied") is False:
        return "record_unsatisfied"
    return "load_docs"


def build_graph():
    graph = StateGraph(InsightState)

    graph.add_node("load_docs", load_documents)
    graph.add_node("build_index", build_vector_index)
    graph.add_node("mem_read", memory_read)
    graph.add_node("retrieve", retrieve_context)
    graph.add_node("answer", generate_answer)
    graph.add_node("infer", infer_intent)
    # graph.add_node("suggest", suggest_action)
    graph.add_node("mem_write", memory_write)
    graph.add_node("record_satisfied", record_satisfied)  # 保存反馈意见的节点
    graph.add_node("record_unsatisfied", record_unsatisfied)  # 保存反馈意见的节点
    graph.add_node("feedback_read", feedback_read)
    graph.add_node("summary_answer", summary_answer)

    graph.add_conditional_edges(
        START,
        decide_where_start,
        {
            "load_docs": "load_docs",
            "record_satisfied": "record_satisfied",
            "record_unsatisfied": "record_unsatisfied"
        }
    )
    graph.add_edge("load_docs", "build_index")
    graph.add_edge("record_unsatisfied", "feedback_read")  # 不满意需要重新生成
    graph.add_edge("build_index", "feedback_read")
    graph.add_edge("feedback_read", "mem_read")  
    graph.add_edge("mem_read", "retrieve")
    graph.add_edge("retrieve", "answer")
    graph.add_edge("answer", "infer")
    graph.add_edge("infer", "mem_write")
    graph.add_edge("mem_write", "summary_answer")
    graph.add_edge("record_satisfied", "summary_answer")
    graph.add_edge("summary_answer", END)

    return graph.compile()
