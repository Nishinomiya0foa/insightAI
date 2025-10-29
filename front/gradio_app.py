import gradio as gr
import requests
import os

API_URL = "http://localhost:8000"

# 上传文档 
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

    resp = requests.post(f"{API_URL}/upload", files=files_payload, data=data)
    return resp.json()


# 发送问题 
def api_ask(session_id, question):
    data = {"session_id": session_id, "question": question}
    resp = requests.post(f"{API_URL}/ask", data=data)
    return resp.json()


# 发送反馈 
def api_feedback(session_id, question, answer, satisfied, new_prompt=None):
    data = {
        "session_id": session_id,
        "question": question,
        "answer": answer,
        "satisfied": satisfied,
        "new_prompt": new_prompt or ""
    }
    resp = requests.post(f"{API_URL}/feedback", json=data)
    return resp.json()


with gr.Blocks() as demo:
    gr.Markdown("# 🤖 InsightAI 自主决策的智能知识体")

    with gr.Tab("📚 文档上传"):
        session_input = gr.Textbox(label="Session ID (为空表示新建知识库和问答)", placeholder="optional")
        files_input = gr.File(file_count="multiple", label="Upload docs (.txt/.md/.pdf/.docx/.xlsx)")
        upload_btn = gr.Button("Upload")
        upload_out = gr.Textbox(label="上传结果 / session_id")

        def do_upload(files, sid):
            if not files:
                return "请选择文件后上传"
            result = api_upload(files, sid)
            if result.get("session_id"):
                return f"Uploaded: {result.get('uploaded')} | session_id: {result.get('session_id')}"
            return str(result)

        upload_btn.click(do_upload, inputs=[files_input, session_input], outputs=upload_out)

    with gr.Tab("💬 智能问答"):
        session_box = gr.Textbox(label="Session ID", placeholder="请输入上方上传返回的 session_id")
        question_box = gr.Textbox(label="输入您的问题", placeholder="基于知识库文档提问")
        ask_btn = gr.Button("Ask")
        answer_box = gr.Textbox(label="Answer", interactive=False)

        # 反馈区域
        with gr.Row(visible=False) as feedback_row:
            gr.Markdown("### 您对本次回答是否满意？")
            satisfied_btn = gr.Button("满意")
            unsatisfied_btn = gr.Button("不满意")

        with gr.Column(visible=False) as feedback_form:
            new_prompt_box = gr.Textbox(label="请说明您的意见或期望的回答方式")
            submit_feedback_btn = gr.Button("提交反馈")
            feedback_result = gr.Textbox(label="反馈结果")

        # 知识问答
        def do_ask(sess, q):
            if not sess:
                return "请输入上传文档后返回的 session_id", gr.update(visible=False), gr.update(visible=False)
            r = api_ask(sess, q)
            text = f"Answer:\n{r.get('answer','')}\n\nIntent:\n{r.get('user_intent','')}\n\nSuggestions:\n{r.get('suggestion','')}"
            return text, gr.update(visible=True), gr.update(visible=False)

        ask_btn.click(do_ask, inputs=[session_box, question_box],
                      outputs=[answer_box, feedback_row, feedback_form])

        # 满意/不满意逻辑
        def on_satisfied(ans):
            return gr.update(visible=False), f"收到您的反馈, 后续将继续以您期望的方式回答问题"

        def on_unsatisfied():
            return gr.update(visible=True)

        satisfied_btn.click(on_satisfied, inputs=[answer_box],
                            outputs=[feedback_form, feedback_result])
        unsatisfied_btn.click(on_unsatisfied, outputs=[feedback_form])

        # 提交反馈
        def send_feedback(sess, q, ans, newp):
            r = api_feedback(sess, q, ans, False, newp)
            text = f"Answer:\n{r.get('answer','')}\n\nIntent:\n{r.get('user_intent','')}\n\nSuggestions:\n{r.get('suggestion','')}"
            return text

        submit_feedback_btn.click(send_feedback,
                                  inputs=[session_box, question_box, answer_box, new_prompt_box],
                                  outputs=[answer_box])

demo.launch()
