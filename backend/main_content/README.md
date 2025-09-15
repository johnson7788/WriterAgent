# 🧠 论文撰写

本项目基于多智能体（Multi-Agent）协作架构，实现从内容大纲出发，自动撰写论文内容的流程。

---

## 🔧 核心功能模块

| Agent 名称                | 功能描述                 |
|-------------------------|----------------------|
---

## 🚀 快速开始

---

### 1. 修改 Agent 使用的模型

编辑模型配置文件以自定义每个 Agent 所调用的模型（如 GPT-4、Claude、Gemini 等）：

```python
# 配置模型的路径，请进行模型修改
backend/slide_agent/slide_agent/config.py
```

---

### 2. 启动本地测试

直接运行多智能体流程测试：

```bash
python main.py
```

---

### 3. 启动后端 API 服务（供前端调用）

提供标准 API 接口（支持 SSE 流式返回），供前端请求：

```bash
python main_api.py
```

## 注意需要修改tools.py中的搜索引擎
backend/slide_agent/slide_agent/tools.py
