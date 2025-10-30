import os
import pandas as pd
import docx
from PyPDF2 import PdfReader
from typing import Dict
import functools
import datetime
import asyncio

def parse_file(file_path: str) -> str:
    """将不同格式文件解析为纯文本"""
    ext = os.path.splitext(file_path)[1].lower()
    text = ""

    if ext == ".txt":
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

    elif ext == ".pdf":
        reader = PdfReader(file_path)
        text = "\n".join([page.extract_text() or "" for page in reader.pages])

    elif ext == ".docx":
        doc = docx.Document(file_path)
        text = "\n".join([para.text for para in doc.paragraphs])

    elif ext in [".xlsx", ".xls"]:
        df = pd.read_excel(file_path)
        text = df.to_string(index=False)

    elif ext == ".csv":
        df = pd.read_csv(file_path)
        text = df.to_string(index=False)

    else:
        text = f"[Unsupported file format: {ext}]"

    return text.strip()


def make_prompt(state: Dict):
    feedbacks = state.get("feedbacks", [])

    q = state.get("question", "")
    context = state.get("context", "")
    prompt = f"根据以下内容回答用户问题：\n{context}\n\n用户问题：{q}\n"
    if feedbacks:
        feedback_desc = "\n".join(feedbacks)
        prompt += f"以下是之前的回答用户不满意的时候提出的意见或期望, 你需要按照用户的想法回答:\n {feedback_desc}"
    prompt += "\n请先回答问题，如果是可执行类的, 可尝试为用户制定markdown格式的任务列表或提醒。"
    return prompt


def log_node_entry(node_name: str = None, desc: str = ""):
    """装饰器：在进入节点时打印日志"""
    def decorator(func):
        name = node_name or func.__name__

        @functools.wraps(func)
        async def async_wrapper(state):
            print(f"\n[{datetime.datetime.now().strftime('%H:%M:%S')}] → [{name}] enter")
            print(f"State keys: {list(state.keys()) if state else []}")
            session_id = state.get("session_id")
            if desc:
                await log_queue_manager.put_log(session_id, f"🟢正在 {desc}")
                if node_name == "summary_answer":
                    await log_queue_manager.put_log(session_id, None)
            result = await func(state)
            print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] → [{name}] completed\n")
            return result
        return async_wrapper
    return decorator


class LogQueueManager:
    def __init__(self):
        self.queues = {}  # {session_id: asyncio.Queue()}

    def get_queue(self, session_id: str):
        if session_id not in self.queues:
            self.queues[session_id] = asyncio.Queue()
        return self.queues[session_id]

    async def put_log(self, session_id: str, message: str):
        queue = self.get_queue(session_id)
        await queue.put(message)

    async def get_log(self, session_id: str):
        queue = self.get_queue(session_id)
        return await queue.get()

    def remove_queue(self, session_id: str):
        if session_id in self.queues:
            del self.queues[session_id]


log_queue_manager = LogQueueManager()
