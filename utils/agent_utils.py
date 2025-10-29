import os
import pandas as pd
import docx
from PyPDF2 import PdfReader
from typing import Dict

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
    feedback = state.get("feedback", "")
    print(2333, feedback)
    if feedback:
        return make_feedback_prompt(state)
    return make_origin_question_prompt(state)


def make_origin_question_prompt(state: Dict):
    q = state.get("question", "")
    context = state.get("context", "")
    prompt = f"根据以下内容回答用户问题：\n{context}\n\n用户问题：{q}\n"
    prompt += "\n请先回答问题，如果是可执行类的, 可尝试为用户制定markdown格式的任务列表或提醒。"
    return prompt


def make_feedback_prompt(state: Dict):
    question = state.get("question", "")
    answer = state.get("answer", "")
    feedback = state.get("feedback", "")
    context = state.get("context", "")

    prompt = f"用户最初的提问是: {question}\n你的回答是: {answer}\n"
    prompt += f"用户对之前的问答的反馈是: {feedback}\n, 请依据以下内容改进你的回答: {context}\n"
    prompt += "\n重新作答后，如果是可执行类的, 可尝试为用户制定markdown格式的任务列表或提醒。"
    return prompt

