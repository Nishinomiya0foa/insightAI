import json, os

MEMORY_FILE = "./memory/session_memory.json"

def load_memory():
    if not os.path.exists(MEMORY_FILE):
        return {"history": []}
    with open(MEMORY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_memory(memory):
    os.makedirs("memory", exist_ok=True)
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory, f, ensure_ascii=False, indent=2)

def add_record(question, answer, suggestion):
    mem = load_memory()
    mem["history"].append({
        "question": question,
        "answer": answer,
        "suggestion": suggestion
    })
    save_memory(mem)
