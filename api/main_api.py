# api/main_api.py
import os, uuid, shutil
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from graph.graph_builder import build_graph
from graph.memory_manager import init_session, append_session
import aiofiles

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
    # 添加到历史
    append_session(session_id, "user", question)
    # langGraph 实现  流程式的问答
    result = GRAPH.invoke(state)
    resp = {
        "session_id": session_id,
        "question": question,
        "context": result.get("context",""),
        "answer": result.get("answer",""),
        "user_intent": result.get("user_intent",""),
        "suggestion": result.get("suggestion",""),
    }
    return JSONResponse(resp)
