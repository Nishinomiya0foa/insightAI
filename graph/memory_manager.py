# core/memory_manager.py
import os, json, time
from typing import Dict, List

MEM_DIR = "memory"
os.makedirs(MEM_DIR, exist_ok=True)

def _mem_path(session_id: str) -> str:
    return os.path.join(MEM_DIR, f"session_{session_id}.json")

def init_session(session_id: str):
    path = _mem_path(session_id)
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"history": []}, f, ensure_ascii=False, indent=2)

def load_session(session_id: str) -> Dict:
    init_session(session_id)
    with open(_mem_path(session_id), "r", encoding="utf-8") as f:
        return json.load(f)

def append_session(session_id: str, role: str, text: str):
    s = load_session(session_id)
    s["history"].append({"role": role, "text": text, "ts": time.time()})
    with open(_mem_path(session_id), "w", encoding="utf-8") as f:
        json.dump(s, f, ensure_ascii=False, indent=2)

def query_session_keywords(session_id: str, query: str, top_k: int=3) -> List[Dict]:
    s = load_session(session_id)
    q_tokens = set([t.lower() for t in query.split() if len(t)>1])
    scored = []
    for rec in s["history"]:
        text = rec.get("text","").lower()
        score = sum(text.count(t) for t in q_tokens)
        scored.append((score, rec))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [rec for score, rec in scored[:top_k] if score>0]
