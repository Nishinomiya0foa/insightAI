
import os
import time
from typing import Dict

from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import JsonOutputParser
from langchain.messages import HumanMessage
from langchain_core.prompts import PromptTemplate

from .memory_manager import (
    init_session, append_session, query_session_keywords, save_feedback_memory, find_last_qa,
    load_feedback_memory)
from config import model, open_api_key, api_base_url, embedding_model
from utils.agent_utils import make_prompt, log_node_entry
from .state_schema import IntentResult


llm = ChatOpenAI(model=model, temperature=0, api_key=open_api_key, openai_api_base=api_base_url)
embeddings = OllamaEmbeddings(model=embedding_model)


@log_node_entry("load_documents", "解析上传的文档")
async def load_documents(state: Dict):
    """加载 session 的上传文档"""
    
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


@log_node_entry("build_vector_index", "构建向量数据索引")
async def build_vector_index(state: Dict):
    """构建或加载 FAISS 索引"""
    
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


@log_node_entry("memory_read", "匹配过往记忆")
async def memory_read(state: Dict):
    """简单关键词匹配记忆"""
    
    session_id = state.get("session_id", "default")
    q = state.get("question", "")
    if not q:
        return {}
    hits = query_session_keywords(session_id, q, top_k=3)
    state["memory_hits"] = hits
    return {"memory_hits": hits}


@log_node_entry("feedback_read", "结合过往反馈")
async def feedback_read(state: Dict):
    """统计历史反馈"""
    
    session_id = state.get("session_id")
    feedbacks = load_feedback_memory(session_id)
    feedbacks = list({d.get("feedback") for d in feedbacks[::-1] if not d.get("satisfied") and d.get("feedback")})[-10:]
    state.update({
        "feedbacks": feedbacks
    })
    
    return {"feedbacks": feedbacks}


@log_node_entry("retrieve_context", "提炼问题相关上下文")
async def retrieve_context(state: Dict):
    """
    根据feedback或question的内容, 获取向量结果
    """
    session_id = state.get("session_id", "default")
    q = state.get("question", "")
    vs_dir = state.get("vector_index_path", os.path.join("data/vectorstore", f"{session_id}_faiss"))
    retrieved = []

    if os.path.exists(vs_dir):
        vectorstore = FAISS.load_local(vs_dir, embeddings, allow_dangerous_deserialization=True)
        docs = await vectorstore.asimilarity_search(q, k=3)
        retrieved = [d.page_content for d in docs]

    if not retrieved:
        for d in state.get("documents", []):
            if q.lower() in d.lower():
                retrieved.append(d[:1000])

    mem_texts = [h["text"] for h in state.get("memory_hits", [])]
    state["retrieved_docs"] = retrieved
    state["context"] = "\n\n".join(retrieved + mem_texts)
    return {"retrieved_docs": retrieved, "context": state["context"]}


@log_node_entry("generate_answer", "生成回答中, 可能用时几十秒")
async def generate_answer(state: Dict):
    """根据上下文生成回答，支持多轮记忆"""
    q = state.get("question", "")
    context = state.get("context", "")
    feedback = state.get("feedback", "")
    
    append_session(state.get("session_id"), "user", q)

    prompt = make_prompt(state)
    messages = [[HumanMessage(content=prompt)]]
    resp = await llm.agenerate(messages)
    answer = resp.generations[0][0].text

    state["answer"] = answer
    state["new_memory_entry"] = {"question": q, "answer": answer, "context": context, "feedback": feedback, "ts": time.time()}
    return {"answer": answer, "new_memory_entry": state["new_memory_entry"]}


@log_node_entry("infer_intent", "推断用户深层需求, 可能用时几十秒")
async def infer_intent(state: Dict):
    """推断用户意图"""
    q = state.get("question", "")
    context = state.get("context", "")
    answer = state.get("answer", "")
    parser = JsonOutputParser(pydantic_object=IntentResult)
    prompt = PromptTemplate(
        template=("""
用户前面的问题：{q}\n
之前的回答:{answer}\n
上下文：{context}\n
根据用户问题、回答和上下文，推断用户的深层意图或真实需求, 预测用户的接下来可能的问题（1-3个）：\n
{intent}
"""),
        input_variables=...,
        partial_variables={"intent": parser.get_format_instructions()}
    )
    chain = prompt | llm | parser
    resp = await chain.ainvoke({"q": q, "answer": answer, "context": context})
    state["user_intent"] = resp.get("intents")
    return {"user_intent": resp.get("intents")}


# @log_node_entry("suggest_action")
# def suggest_action(state: Dict):
#     """生成行动建议"""
#     intent = state.get("user_intent", "")
#     prompt = f"请基于以下推测的用户需求给出简洁但可执行的建议：\n{intent}"

#     messages = [[HumanMessage(content=prompt)]]
#     resp = llm.generate(messages)
#     suggestion = resp.generations[0][0].text
#     state["suggestion"] = suggestion
#     return {"suggestion": suggestion}


@log_node_entry("memory_write")
async def memory_write(state: Dict):
    """写入新记忆"""
    session_id = state.get("session_id", "default")
    entry = state.get("new_memory_entry")
    if entry:
        append_session(session_id, "assistant", entry.get("answer", ""))
        append_session(session_id, "system", f"suggestion:{state.get('suggestion','')}")
    return {}


def record_feedback(state: Dict):
    session_id = state.get("session_id")
    last_q, last_a = find_last_qa(session_id)
    
    if not last_q:
        return False
    save_feedback_memory(
        dict(
            session_id=session_id,
            question=last_q.get("text"),
            answer=last_a.get("text"),
            satisfied = state.get("satisfied"),
            feedback=state.get("feedback")
        )
    )
    state.update({
        "question": last_q.get("text"),
        "answer": last_a.get("text"),
        "feedback": state.get("feedback")
    })
    
    return {
        "question": last_q.get("text"),
        "answer": last_a.get("text"),
        "feedback": state.get("feedback")
    }


@log_node_entry("record_satisfied", "记录用户的反馈")
async def record_satisfied(state: Dict):
    """记录<满意>的反馈"""
    return record_feedback(state)


@log_node_entry("record_unsatisfied", "记录用户的反馈")
async def record_unsatisfied(state: Dict):
    """记录<不满意>的反馈"""
    return record_feedback(state)


@log_node_entry("summary_answer", "总结汇总完整回答")
async def summary_answer(state: Dict):
    """总结汇总回答, 目前没实际作用, 仅用来统一graph出口"""
    return {}

