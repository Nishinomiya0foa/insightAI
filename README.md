# InsightAI

**InsightAI: 智能知识推理与反馈系统**  
基于 LangChain 和 LangGraph 的多文档智能问答与自主推理 Agent，支持文档解析、上下文问答、多步推理、用户反馈与实时日志追踪。

---

## 功能特色

- **多文档知识管理**  
  支持 PDF、Word、Excel、Markdown 等多格式文档上传与解析。

- **智能问答与推理**  
  基于文档上下文与 GPT-4/5 模型进行多步逻辑推理和结构化回答。

- **探索式对话**  
  回答后自动生成 1-3 个后续问题，引导用户深入探索。

- **用户反馈与自我优化**  
  满意/不满意反馈可触发再学习机制，逐步优化回答质量。

- **实时执行日志**  
  节点级执行进度异步推送，前端可实时可视化思考链。

---

## 技术栈
- Python 3.11+
- FastAPI
- LangChain / LangGraph
- OpenAI GPT-4/5
- FAISS 向量检索
- Gradio 前端
- PyMuPDF / python-docx / pandas

## 快速开始
1. 克隆仓库
```bash
git clone https://github.com/Nishinomiya0foa/insightAI.git
cd insightAI
```
2. 安装依赖
```bash
pip install -r requirements.txt
```
3. 配置config.py
4. 启动服务
```bash
uvicorn api.main_api:app --reload
```
5. 启动前端
```bash
python front/gradio_app.py
```
6. 浏览器地址
```
http://localhost:7860
```

## 项目亮点

- 步节点日志队列，支持多用户会话隔离

- 多轮探索式对话，提升用户交互体验

- 可扩展 LangGraph 节点，支持自定义推理逻辑

- 反馈与记忆模块，初步实现反思-再计划能力


