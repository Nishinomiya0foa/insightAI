import gradio as gr
import requests
import os

API_URL = "http://localhost:8001"

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
def api_feedback(session_id, satisfied, new_prompt=None):
    data = {
        "session_id": session_id,
        "satisfied": satisfied,
        "feedback": new_prompt or ""
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
        progress_box = gr.Textbox(label="思考进度", interactive=False)

        intent_dropdown = gr.Radio(choices=[], label="选择意图", interactive=True, visible=False)
        intent_ask_btn = gr.Button("追问", visible=False)

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
            text = f"Answer:\n{r.get('answer','')}\n\nIntent:\n{r.get('user_intent',[])}\n\n"
            return (
                q, text, gr.update(visible=True), gr.update(visible=False), 
                gr.update(choices=r.get('user_intent',[]), value=None, visible=bool(r.get('user_intent',[]))),
                gr.update(visible=True)
            )
        
        def get_progress(sess):
            resp = requests.get(f"{API_URL}/get_progress?session_id={sess}", stream=True)
            yield gr.update(value="")
            # 用于保存日志的变量
            result = ""
            for line in resp.iter_lines():
                if line:
                    chunk = line.decode("utf-8")
                    result += chunk + "\n"
                    yield result  # 每次更新

        ask_btn.click(do_ask, inputs=[session_box, question_box],
                      outputs=[question_box, answer_box, feedback_row, feedback_form, intent_dropdown, intent_ask_btn])
        ask_btn.click(get_progress, inputs=[session_box,],
                      outputs=[progress_box])

        # 满意逻辑
        def on_satisfied(sess):
            r = api_feedback(sess, True)
            return gr.update(visible=True), f"收到您的反馈, 将根据您的喜好调整回答"

        # 不满意逻辑
        def on_unsatisfied():
            return gr.update(visible=True), f"收到您的反馈, 将根据您的喜好调整回答"

        # 提交反馈
        def send_feedback(sess, newp):
            r = api_feedback(sess, False, newp)
            text = f"Answer:\n{r.get('answer','')}\n\nIntent:\n{r.get('user_intent',[])}\n\n"
            return text
        satisfied_btn.click(on_satisfied, inputs=[session_box, ],
                            outputs=[feedback_form, feedback_result])
        unsatisfied_btn.click(on_unsatisfied, outputs=[feedback_form, feedback_result])
        unsatisfied_btn.click(get_progress, inputs=[session_box,],
                      outputs=[progress_box])
        submit_feedback_btn.click(send_feedback,
                                  inputs=[session_box, new_prompt_box],
                                  outputs=[answer_box])

        intent_ask_btn.click(do_ask, inputs=[session_box, intent_dropdown],
                      outputs=[question_box, answer_box, feedback_row, feedback_form, intent_dropdown, intent_ask_btn])
        intent_ask_btn.click(get_progress, inputs=[session_box,],
                      outputs=[progress_box])

demo.launch()
