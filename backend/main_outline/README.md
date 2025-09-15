# 用于生成PPT大纲

本项目结合了 **A2A 框架** 与 **Google ADK 架构**，通过 SSE（Server-Sent Events）生成大纲


## 📂 项目结构说明

| 文件                      | 说明                                |
| ----------------------- | --------------------------------- |
| `a2a_client.py`         | A2A 客户端封装，管理与服务端的交互               |
| `agent.py`              | 主控制 Agent，调度执行任务                  |
| `main_api.py`           | FastAPI 主入口，暴露 SSE 接口             |
| `adk_agent_executor.py` | 实现 A2A 与 ADK 框架的集成逻辑              |
| `.env`                  | 环境变量配置，需包含 Google GenAI API 密钥等信息 |


# 注意：
main_api.py服务器部署时的agent_url，是对外提供服务的url，有时可能和监听地址不同，尤其是内外网环境时。

## todo
1. 容器化
2. 搜索工具