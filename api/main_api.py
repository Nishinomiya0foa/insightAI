# api/main_api.py
import os, uuid, shutil
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from graph.graph_builder import build_graph
from graph.memory_manager import init_session, append_session, save_feedback_memory
import aiofiles
from utils.agent_utils import log_queue_manager

from .param_schema import FeedbackRequest

app = FastAPI(title="InsightAI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

GRAPH = build_graph()

@app.post("/upload")
async def upload_files(session_id: str = Form(None), files: list[UploadFile] = File(...)):
    """
    支持上传多个文件, 且支持增量上传, 以session_id为key
    """
    if session_id is None:
        session_id = str(uuid.uuid4())[:8]
    dest_dir = os.path.join("data/uploaded_files", session_id)
    os.makedirs(dest_dir, exist_ok=True)
    for f in files:
        file_path = os.path.join(dest_dir, f.filename)
        async with aiofiles.open(file_path, "wb") as out:
            content = await f.read()
            await out.write(content)
    # init memory session
    init_session(session_id)
    return JSONResponse({"ok": True, "session_id": session_id, "uploaded": [f.filename for f in files]})

@app.post("/ask")
async def ask_question(session_id: str = Form(...), question: str = Form(...)):
    """
    在session环境下, 提问 获取 回答
    """
    state = {"session_id": session_id, "question": question}
    # langGraph 实现  流程式的问答
    result = await GRAPH.ainvoke(state)
    resp = {
        "session_id": session_id,
        "question": question,
        "context": result.get("context",""),
        "answer": result.get("answer",""),
        "user_intent": result.get("user_intent",""),
        "suggestion": result.get("suggestion",""),
    }
    return JSONResponse(resp)


@app.get("/get_progress")
async def get_progress(session_id):
    # 返回流式日志响应
    async def event_stream():
        queue = log_queue_manager.get_queue(session_id)
        while True:
            log = await queue.get()  # 从队列获取日志
            if log is None:
                log_queue_manager.remove_queue(session_id)
                break
            yield log + "\n"  # 每次 yield 返回一个新的日志行

    return StreamingResponse(event_stream(), media_type="text/plain")


@app.post("/feedback")
async def user_feedback(item: FeedbackRequest):
    """
    用户提交反馈：
    - satisfied: True / False
    - 如果不满意并提供 new_prompt，将重新生成答案
    """
    state = {"session_id": item.session_id, "feedback": item.feedback, "satisfied": item.satisfied}
    regenerated_answer = await GRAPH.ainvoke(state)

    resp = {
        "session_id": item.session_id,
        "question": regenerated_answer.get("question", ""),
        "context": regenerated_answer.get("context",""),
        "answer": regenerated_answer.get("answer",""),
        "user_intent": regenerated_answer.get("user_intent",[]),
        # "suggestion": regenerated_answer.get("suggestion",""),
    }
    return JSONResponse(resp)
