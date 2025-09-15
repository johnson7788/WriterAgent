import asyncio
import json
import re
import os
import dotenv
import uuid
import httpx
from fastapi import FastAPI
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from outline_client import A2AOutlineClientWrapper
from content_client import A2AContentClientWrapper
dotenv.load_dotenv()

"""
后端:
- 仅生成“综述类论文”的 Markdown；**不再返回 JSON 结构**，也**不再提供静态资源**。
- 路由改为：
  - POST /api/review_outline  —— 生成大纲（Markdown，支持流式）
  - POST /api/review          —— 生成正文章节（Markdown，支持流式）；可按大纲逐章生成
- 典型用法：先调 /api/review_outline 得到大纲，再根据所选章节标题/序号逐个调 /api/review。
"""
OUTLINE_API = os.environ["OUTLINE_API"]
CONTENT_API = os.environ["CONTENT_API"]
app = FastAPI(title="Review Paper Generator (Markdown, Mock)", version="1.1.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ReviewOutlineRequest(BaseModel):
    topic: str = Field(..., description="综述主题，例如 '多模态大模型安全性'")
    language: str = Field("zh", description="'zh' 或 'en'")

class ReviewChapterRequest(BaseModel):
    topic: str = Field(..., description="综述主题")
    outline: str|dict = Field(..., description="大纲数组。如果不传，使用默认模板大纲")
    language: str = Field("zh", description="'zh' 或 'en'")


async def stream_outline_response(prompt: str):
    """A generator that yields parts of the agent response."""
    outline_wrapper = A2AOutlineClientWrapper(session_id=uuid.uuid4().hex, agent_url=OUTLINE_API)
    async for chunk_data in outline_wrapper.generate(prompt):
        print(f"生成大纲输出的chunk_data: {chunk_data}")
        yield "data: " + json.dumps(chunk_data,ensure_ascii=False) + "\n\n"

@app.post("/api/outline")
async def api_review_outline(req: ReviewOutlineRequest):
    topic = (req.topic or "示例主题").strip()
    return StreamingResponse(stream_outline_response(topic), media_type="text/event-stream; charset=utf-8")

class AipptContentRequest(BaseModel):
    content: str

async def stream_content_response(markdown_content: str):
    """  # PPT的正文内容生成"""
    # 用正则找到第一个一级标题及之后的内容
    match = re.search(r"(# .*)", markdown_content, flags=re.DOTALL)

    if match:
        result = markdown_content[match.start():]
    else:
        result = markdown_content
    print(f"用户输入的markdown大纲是：{result}")
    content_wrapper = A2AContentClientWrapper(session_id=uuid.uuid4().hex, agent_url=CONTENT_API)
    async for chunk_data in content_wrapper.generate(result):
        print(f"生成文章内容的输出的chunk_data: {chunk_data}")
        yield "data: " + json.dumps(chunk_data,ensure_ascii=False) + "\n\n"
@app.post("/api/review")
async def api_review(req: ReviewChapterRequest):
    topic = (req.topic or "示例主题").strip()
    outline = req.outline
    return StreamingResponse(stream_content_response(outline), media_type="text/event-stream; charset=utf-8")

class GenerateParagraphRequest(BaseModel):
    prompt: str = Field(..., description="要进行改写或者生成的提示词，例如某个段落")
    option: str = Field(..., description="要进行操作的命令，根据不同的操作要求，和prompt进行更多生成")

@app.post("/api/pragraph")
async def api_pragraph(req: GenerateParagraphRequest):
    print(f"进行改写或者生成，prompt是：{req.prompt}")

@app.get("/ping")
async def Ping():
    return "PONG"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=7800)