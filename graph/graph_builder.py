# core/graph_builder.py
from langgraph.graph import StateGraph, START, END
from .state_schema import InsightState
from .nodes import (
    load_documents, build_vector_index, memory_read,
    retrieve_context, generate_answer, infer_intent,
    suggest_action, memory_write
)

def build_graph():
    graph = StateGraph(InsightState)

    graph.add_node("load_docs", load_documents)
    graph.add_node("build_index", build_vector_index)
    graph.add_node("mem_read", memory_read)
    graph.add_node("retrieve", retrieve_context)
    graph.add_node("answer", generate_answer)
    graph.add_node("infer", infer_intent)
    graph.add_node("suggest", suggest_action)
    graph.add_node("mem_write", memory_write)

    graph.add_edge(START, "load_docs")
    graph.add_edge("load_docs", "build_index")
    graph.add_edge("build_index", "mem_read")
    graph.add_edge("mem_read", "retrieve")
    graph.add_edge("retrieve", "answer")
    graph.add_edge("answer", "infer")
    graph.add_edge("infer", "suggest")
    graph.add_edge("suggest", "mem_write")
    graph.add_edge("mem_write", END)

    return graph.compile()
