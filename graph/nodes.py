
import os
import time
from typing import Dict

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.messages import HumanMessage

from .memory_manager import init_session, append_session, query_session_keywords
from config import model, open_api_key, api_base_url, embedding_model
from utils.agent_utils import make_prompt
from .node_helper import make_prompt_from_feedback_memory


llm = ChatOpenAI(model=model, temperature=0, api_key=open_api_key, openai_api_base=api_base_url)
embeddings = OllamaEmbeddings(model=embedding_model)


def load_documents(state: Dict):
    """加载 session 的上传文档"""
    print("load doc")
    session_id = state.get("session_id", "default")
    data_dir = os.path.join("data/uploaded_files", session_id)
    docs = []
    if os.path.exists(data_dir):
        for fn in os.listdir(data_dir):
            if fn.lower().endswith((".txt", ".md")):
                with open(os.path.join(data_dir, fn), "r", encoding="utf-8") as f:
                    docs.append(f.read())
    state["documents"] = docs
    init_session(session_id)
    return {"documents": docs}


def build_vector_index(state: Dict):
    """构建或加载 FAISS 索引"""
    print("build vector")
    session_id = state.get("session_id", "default")
    vs_dir = os.path.join("data/vectorstore", f"{session_id}_faiss")
    docs = state.get("documents", [])
    if not docs:
        return {}

    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    chunks = []
    for d in docs:
        chunks.extend(splitter.split_text(d))

    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
    os.makedirs(vs_dir, exist_ok=True)
    vectorstore.save_local(vs_dir)

    state["vector_index_path"] = vs_dir
    return {"vector_index_path": vs_dir}


def memory_read(state: Dict):
    """简单关键词匹配记忆"""
    print(f"mem read")
    session_id = state.get("session_id", "default")
    # 如果有用户的feedback, 只用匹配feedback的命中, 原question的回答会直接给llm
    q = state.get("feedback", "") or state.get("question", "")
    if not q:
        return {}
    hits = query_session_keywords(session_id, q, top_k=3)
    state["memory_hits"] = hits
    return {"memory_hits": hits}


def retrieve_context(state: Dict):
    """
    根据feedback或question的内容, 获取向量结果
    """
    session_id = state.get("session_id", "default")
    q = state.get("feedback", "") or state.get("question", "")
    vs_dir = state.get("vector_index_path", os.path.join("data/vectorstore", f"{session_id}_faiss"))
    retrieved = []

    if os.path.exists(vs_dir):
        vectorstore = FAISS.load_local(vs_dir, embeddings, allow_dangerous_deserialization=True)
        docs = vectorstore.similarity_search(q, k=3)
        retrieved = [d.page_content for d in docs]

    if not retrieved:
        for d in state.get("documents", []):
            if q.lower() in d.lower():
                retrieved.append(d[:1000])

    mem_texts = [h["text"] for h in state.get("memory_hits", [])]
    state["retrieved_docs"] = retrieved
    state["context"] = "\n\n".join(retrieved + mem_texts)
    return {"retrieved_docs": retrieved, "context": state["context"]}


def generate_answer(state: Dict):
    """根据上下文生成回答，支持多轮记忆"""
    q = state.get("question", "")
    context = state.get("context", "")
    feedback = state.get("feedback", "")

    prompt = make_prompt(state)
    # 从历史反馈中提取偏好
    prompt += make_prompt_from_feedback_memory(state.get("session_id"))

    messages = [[HumanMessage(content=prompt)]]
    resp = llm.generate(messages)
    answer = resp.generations[0][0].text

    state["answer"] = answer
    state["new_memory_entry"] = {"question": q, "answer": answer, "context": context, "feedback": feedback, "ts": time.time()}
    return {"answer": answer, "new_memory_entry": state["new_memory_entry"]}


def infer_intent(state: Dict):
    """推断用户意图"""
    q = state.get("feedback", "") or state.get("question", "")
    context = state.get("context", "")
    prompt = f"根据用户问题和上下文，推断用户的深层意图或真实需求, 或预测用户的下一步问题（1-2 句）：\n问题：{q}\n上下文：{context}"

    messages = [[HumanMessage(content=prompt)]]
    resp = llm.generate(messages)
    intent = resp.generations[0][0].text
    state["user_intent"] = intent
    return {"user_intent": intent}


def suggest_action(state: Dict):
    """生成行动建议"""
    intent = state.get("user_intent", "")
    prompt = f"请基于以下推测的用户需求给出简洁但可执行的建议：\n{intent}"

    messages = [[HumanMessage(content=prompt)]]
    resp = llm.generate(messages)
    suggestion = resp.generations[0][0].text
    state["suggestion"] = suggestion
    return {"suggestion": suggestion}


def memory_write(state: Dict):
    """写入新记忆"""
    session_id = state.get("session_id", "default")
    entry = state.get("new_memory_entry")
    if entry:
        append_session(session_id, "assistant", entry.get("answer", ""))
        append_session(session_id, "system", f"suggestion:{state.get('suggestion','')}")
    return {}
