
import os, json, time
from typing import Dict, List

MEM_DIR = "data/memory"
os.makedirs(MEM_DIR, exist_ok=True)
FEEDBACK_DIR = "data/feedback_memory"
os.makedirs(FEEDBACK_DIR, exist_ok=True)

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


def _get_memory_path(session_id: str) -> str:
    """根据 session_id 生成对应记忆文件路径"""
    filename = f"feedback_{session_id}.json"
    return os.path.join(FEEDBACK_DIR, filename)


def save_feedback_memory(entry: Dict):
    """
    将反馈存入指定记忆文件中。
    """
    session_id = entry.get("session_id", "default_session")
    path = _get_memory_path(session_id)

    # 读取原有数据
    data: List[Dict] = []
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            data = []

    # 写入新记录
    entry["ts"] = time.time()
    data.append(entry)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_feedback_memory(session_id: str) -> List[Dict]:
    """加载全部记忆反馈"""
    path = _get_memory_path(session_id)
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return []


def get_recent_feedbacks(session_id: str, limit: int = 3) -> List[Dict]:
    """获取最近几条反馈记录"""
    data = load_feedback_memory(session_id)
    return data[-limit:] if data else []
