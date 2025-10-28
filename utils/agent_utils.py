import os
import pandas as pd
import docx
from PyPDF2 import PdfReader

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
