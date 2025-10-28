from langchain_openai import ChatOpenAI




import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from .memory_manager import add_record, load_memory

open_api_key = "sk-5efbb49050284a6d936530674f1cb22f"
model = "deepseek-chat"
api_base_url = "https://api.deepseek.com"
llm = ChatOpenAI(model=model, temperature=0, api_key=open_api_key, openai_api_base=api_base_url)
embeddings = OllamaEmbeddings(model="llama3-groq-tool-use:latest")

def load_documents(state):
    docs_dir = "data"
    docs = []
    for filename in os.listdir(docs_dir):
        if filename.endswith(".txt"):
            with open(os.path.join(docs_dir, filename), "r", encoding="utf-8") as f:
                docs.append(f.read())
    state["documents"] = docs
    state["memory"] = load_memory()
    return state


def build_vector_index(state):
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    all_chunks = []
    for doc in state["documents"]:
        all_chunks.extend(splitter.split_text(doc))

    vectorstore = FAISS.from_texts(all_chunks, embedding=embeddings)

    os.makedirs("vectorstore/faiss_index", exist_ok=True)
    vectorstore.save_local("vectorstore/faiss_index")

    state["vectorstore"] = vectorstore
    return state


def retrieve_context(state):
    vectorstore = state.get("vectorstore") or FAISS.load_local(
        "vectorstore/faiss_index", embeddings, allow_dangerous_deserialization=True
    )
    question = state["question"]
    results = vectorstore.similarity_search(question, k=3)
    retrieved = [r.page_content for r in results]
    state["retrieved_docs"] = retrieved
    state["context"] = "\n\n".join(retrieved)
    return state


def generate_answer(state):
    context = state["context"]
    question = state["question"]
    prompt = f"根据以下内容回答问题：\n\n{context}\n\n问题：{question}"
    answer = llm.invoke(prompt)
    state["answer"] = answer.content
    return state


def infer_intent(state):
    """推断用户的潜在需求"""
    question = state["question"]
    context = state["context"]
    prompt = f"""
你是一个洞察型AI，请根据用户问题和上下文内容，推测用户可能的真实意图或需求。
用户问题：{question}
相关上下文：{context}
请简要说明推测的用户需求。
"""
    intent = llm.invoke(prompt)
    state["user_intent"] = intent.content.strip()
    return state


def suggest_action(state):
    """根据推测意图，提供可执行建议"""
    intent = state["user_intent"]
    prompt = f"""
你是一个智能决策顾问。根据以下意图，提供3条具体的行动或改进建议：
{intent}
"""
    suggestion = llm.invoke(prompt)
    state["suggestion"] = suggestion.content.strip()
    return state


def save_memory_node(state):
    add_record(
        question=state["question"],
        answer=state["answer"],
        suggestion=state["suggestion"]
    )
    return state

