import json
import asyncio
import os.path
from datetime import datetime
from typing import List, Optional
from convert_outline import parse_outline_markdown_to_json
from fastapi import FastAPI
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse

"""
后端改造点（模拟数据）：
- 仅生成“综述类论文”的 Markdown；**不再返回 JSON 结构**，也**不再提供静态资源**。
- 路由改为：
  - POST /api/review_outline  —— 生成大纲（Markdown，支持流式）
  - POST /api/review          —— 生成正文章节（Markdown，支持流式）；可按大纲逐章生成
- 典型用法：先调 /api/review_outline 得到大纲，再根据所选章节标题/序号逐个调 /api/review。
"""

app = FastAPI(title="Review Paper Generator (Markdown, Mock)", version="1.1.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================================
# 请求模型
# ================================
class ReviewOutlineRequest(BaseModel):
    topic: str = Field(..., description="综述主题，例如 '多模态大模型安全性'")
    language: str = Field("zh", description="'zh' 或 'en'")

class ReviewChapterRequest(BaseModel):
    topic: str = Field(..., description="综述主题")
    outline: str|dict = Field(..., description="大纲数组。如果不传，使用默认模板大纲")
    language: str = Field("zh", description="'zh' 或 'en'")

# ================================
# 大纲模板与生成
# ================================
# ================================
# 章节内容生成（模拟）
# ================================
OUTLINE_DATA = []
if os.path.exists("outline_data.txt"):
    with open("outline_data.txt", "r") as f:
        outline_content = f.read()
    for line in outline_content.split("\n"):
        if line.startswith("data:"):
            line = line[5: ]
            one_data = json.loads(line)
            OUTLINE_DATA.append(one_data)
    print(f"共收集了{len(OUTLINE_DATA)}条数据")

BODY_SECONTION_CONTENTS = []
if os.path.exists("body_data.txt"):
    with open("body_data.txt", "r") as f:
        body_content = f.read()
    for line in body_content.split("\n"):
        if line.startswith("data:"):
            line = line[5: ]
            one_data = json.loads(line)
            BODY_SECONTION_CONTENTS.append(one_data)
    print(f"共收集了{len(BODY_SECONTION_CONTENTS)}条数据")

def build_outline_markdown(topic: str, lang: str) -> str:
    return OUTLINE_DATA

# ================================
# 流式工具
# ================================
async def outline_stream(outline_data: str, delay: float = 0.002):
    for one_data in outline_data:
        yield "data: " + json.dumps(one_data, ensure_ascii=False) + "\n\n"
        await asyncio.sleep(delay)
# ================================
# 路由：大纲（Markdown）
# ================================
@app.post("/api/outline")
async def api_review_outline(req: ReviewOutlineRequest):
    topic = (req.topic or "示例主题").strip()
    return StreamingResponse(outline_stream(outline_data=OUTLINE_DATA), media_type="text/event-stream; charset=utf-8")

# ================================
# 路由：正文（按大纲逐章生成，Markdown）
# ================================
async def body_stream(sections: list, delay: float = 0.01):
    # 正文的内容
    for section_content in sections:
        yield "data: " + json.dumps(section_content, ensure_ascii=False) + "\n\n"
        await asyncio.sleep(delay)
@app.post("/api/review")
async def api_review(req: ReviewChapterRequest):
    topic = (req.topic or "示例主题").strip()
    outline = req.outline
    sections = BODY_SECONTION_CONTENTS
    # 选择目标章节或整篇
    return StreamingResponse(body_stream(sections=sections), media_type="text/event-stream; charset=utf-8")

# ================================
# 根路由与健康检查
# ================================
@app.get("/")
async def root():
    return {
        "name": "Review Paper Generator (Markdown, Mock)",
        "version": "1.1.0",
        "endpoints": {
            "POST /api/review_outline": "生成大纲（Markdown/流式）",
            "POST /api/review": "生成正文（按大纲单章或整篇，Markdown/流式）",
        },
    }

class GenerateParagraphRequest(BaseModel):
    prompt: str = Field(..., description="要进行改写或者生成的提示词，例如某个段落")
    option: str = Field(..., description="要进行操作的命令，根据不同的操作要求，和prompt进行更多生成")

@app.post("/api/pragraph")
async def api_pragraph(req: GenerateParagraphRequest):
    print(f"进行改写或者生成，prompt是：{req.prompt}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=7800)
