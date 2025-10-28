from langgraph.graph import StateGraph
from .state_schema import InsightState
from .nodes import (
    load_documents, build_vector_index, retrieve_context,
    generate_answer, infer_intent, suggest_action, save_memory_node
)

def build_graph():
    graph = StateGraph(InsightState)
    graph.add_node("load_docs", load_documents)
    graph.add_node("build_index", build_vector_index)
    graph.add_node("retrieve", retrieve_context)
    graph.add_node("answer", generate_answer)
    graph.add_node("infer", infer_intent)
    graph.add_node("suggest", suggest_action)
    graph.add_node("save_memory", save_memory_node)

    graph.add_edge("load_docs", "build_index")
    graph.add_edge("build_index", "retrieve")
    graph.add_edge("retrieve", "answer")
    graph.add_edge("answer", "infer")
    graph.add_edge("infer", "suggest")
    graph.add_edge("suggest", "save_memory")

    graph.set_entry_point("load_docs")
    graph.set_finish_point("save_memory")

    return graph.compile()
