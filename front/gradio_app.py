# frontend/gradio_app.py
import gradio as gr
import requests
import os

API_URL = "http://localhost:8000"

def api_upload(files, session_id):
    if not files:
        return {"error": "未选择任何文档"}

    files_payload = []
    for path in files:
        filename = os.path.basename(path)
        with open(path, "rb") as f:
            files_payload.append(("files", (filename, f.read(), "application/octet-stream")))

    data = {}
    if session_id:
        data["session_id"] = session_id

    import requests
    resp = requests.post(f"{API_URL}/upload", files=files_payload, data=data)
    return resp.json()

def api_ask(session_id, question):
    data = {"session_id": session_id, "question": question}
    resp = requests.post(f"{API_URL}/ask", data=data)
    return resp.json()

with gr.Blocks() as demo:
    gr.Markdown("# InsightAI 自主决策的智能知识体")

    with gr.Row():
        session_input = gr.Textbox(label="Session ID (为空表示新建知识库和问答)", placeholder="optional")
        files_input = gr.File(file_count="multiple", label="Upload docs (.txt/.md)")

        upload_btn = gr.Button("Upload")
        upload_out = gr.Textbox(label="知识库文档和session id")

    with gr.Row():
        session_box = gr.Textbox(label="Session ID", value="")
        question = gr.Textbox(label="输入您的问题", placeholder="基于知识库文档提问")
        ask_btn = gr.Button("Ask")
        answer_box = gr.Textbox(label="Answer")

    def do_upload(files, sid):
        if not files:
            return "请选择文件后上传"
        result = api_upload(files, sid)
        if result.get("session_id"):
            return f"Uploaded: {result.get('uploaded')}  | session_id: {result.get('session_id')}"
        return str(result)

    upload_btn.click(do_upload, inputs=[files_input, session_input], outputs=upload_out)

    def do_ask(sess, q):
        if not sess:
            return "请输入上传文档后返回的session_id"
        r = api_ask(sess, q)
        text = f"Answer:\n{r.get('answer','')}\n\nIntent:\n{r.get('user_intent','')}\n\nSuggestions:\n{r.get('suggestion','')}"
        return text

    ask_btn.click(do_ask, inputs=[session_box, question], outputs=answer_box)

demo.launch()
