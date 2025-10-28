from graph.graph_builder import build_graph

if __name__ == "__main__":
    app = build_graph()
    state = {
        "question": "准备web框架和langchain的面试"
    }
    result = app.invoke(state)

    print("\n 用户问题：", result["question"])
    # print("\n 检索内容：\n", result["context"])
    print("\n 回答：\n", result["answer"])
    print("\n 推测意图：\n", result["user_intent"])
    print("\n 行动建议：\n", result["suggestion"])

    MEMORY_FILE = "./memory/last_record.json"
    import json
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)