import asyncio
import json
import os
import uuid
import hashlib
import time
from typing import Any, Dict, List, Optional
from pathlib import Path

import dotenv
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

dotenv.load_dotenv()

"""
包含的API：
1) POST /api/search_keyword
   - 正常搜索：返回 {id, file_id, title, snippet, score, publish_time, url}
     并写入缓存（内存 + 本地文件 file_id.json）
   - 若传入 file_ids：按ID返回正文 {file_id, title, content, publish_time, url}
2) GET /api/article/{file_id}
   - 通过 file_id 获取完整正文（先内存，缺失则回读本地文件）
环境变量：
- ARTICLE_STORE_DIR：缓存文件目录（默认 ./article_cache）
"""

# ======== 依赖你现有的 weixin_search.py ========
from weixin_search import get_wechat_article
from zhipu_search import zhipu_search_web

app = FastAPI(title="Search API tool", version="1.3.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===================== 配置：本地缓存目录 =====================
DATA_DIR = Path(os.getenv("ARTICLE_STORE_DIR", "./article_cache")).resolve()
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ===================== 数据模型 =====================
class KeywordSearchRequest(BaseModel):
    keyword: str = Field(..., min_length=1, description="搜索关键词")
    limit: int = Field(10, ge=1, le=50, description="返回条数上限，默认10")
    file_ids: List[str] = Field([], description="多个文件id，可选传入（若传则走按id取内容）")
    search_engine: str = Field("default", description="使用的搜索引擎,例如zhipu, weixin")

class ArticleHit(BaseModel):
    id: str
    file_id: str
    title: str
    snippet: str
    score: float = 1.0
    publish_time: Optional[str] = None
    url: Optional[str] = None

class ArticleDoc(BaseModel):
    file_id: str
    title: str
    content: str
    publish_time: Optional[str] = None
    url: Optional[str] = None

# ===================== 内存缓存（热数据） =====================
# 结构：{ file_id: { "title":..., "publish_time":..., "url":..., "content":..., "created_at": ts } }
ARTICLE_CACHE: Dict[str, Dict[str, Any]] = {}

# ===================== 工具方法：file_id、片段、磁盘IO =====================
def compute_file_id(title: str, publish_time: Optional[str] = None) -> str:
    """
    用 title|publish_time 生成更稳的 file_id，避免同名标题碰撞。
    返回 16位短sha1。
    """
    key = f"{title.strip()}|{(publish_time or '').strip()}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]

def _make_snippet(text: str, length: int = 160) -> str:
    text = (text or "").strip().replace("\r", "").replace("\n", " ")
    return text[:length] + ("…" if len(text) > length else "")

def _file_path(file_id: str) -> Path:
    return DATA_DIR / f"{file_id}.json"

def _save_to_disk(file_id: str, record: Dict[str, Any]) -> None:
    """
    原子写入：先写 .tmp，再替换为 .json
    - 否则保存包含元数据的JSON
    """
    path = _file_path(file_id)
    tmp = path.with_suffix(".json.tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)

    # 含元数据
    payload = {
        "title": record.get("title", ""),
        "publish_time": record.get("publish_time"),
        "url": record.get("url"),
        "content": record.get("content", ""),
        "created_at": record.get("created_at", time.time()),
    }

    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)

    tmp.replace(path)

def _load_from_disk(file_id: str) -> Optional[Dict[str, Any]]:
    """
    从磁盘读取并转换为内存结构，若不存在返回 None。
    """
    path = _file_path(file_id)
    if not path.exists():
        return None

    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return {
            "title": obj.get("title", ""),
            "publish_time": obj.get("publish_time"),
            "url": obj.get("url"),
            "content": obj.get("content", ""),
            "created_at": obj.get("created_at", path.stat().st_mtime),
        }
    except Exception:
        return None

def upsert_cache(file_id: str, *, title: str, publish_time: Optional[str], url: Optional[str], content: str) -> None:
    record = {
        "title": title,
        "publish_time": publish_time,
        "url": url,
        "content": content,
        "created_at": time.time(),
    }
    ARTICLE_CACHE[file_id] = record
    # 写入磁盘
    _save_to_disk(file_id, record)

def get_from_cache(file_id: str) -> Optional[Dict[str, Any]]:
    """
    优先读内存；若内存没有，则尝试从磁盘读并回填内存。
    """
    r = ARTICLE_CACHE.get(file_id)
    if r:
        return r
    r = _load_from_disk(file_id)
    if r:
        ARTICLE_CACHE[file_id] = r
    return r

# ===================== 搜索实现 =====================
def _search_articles_sync(keyword: str, limit: int, search_engine: str="default") -> List[ArticleHit]:
    """
    同步函数：调用 weixin 搜索并填充缓存（内存 + 磁盘），返回 ArticleHit 列表。
    """
    if search_engine == "default":
        search_engine = os.environ["USE_WEB_SEARCH"]
    if search_engine == "weixin":
        print(f"使用的搜索引擎是： weixin")
        result = get_wechat_article(keyword=keyword, number=limit)
    else:
        print(f"使用的搜索引擎是： zhipu")
        status, result = zhipu_search_web(keyword=keyword, number=limit)
    print(f"使用了搜索工具的搜索结果是: {result}")
    if isinstance(result, str):
        return []
    hits: List[ArticleHit] = []
    for item in result:
        title = item.get("title") or ""
        publish_time = item.get("publish_time")
        url = item.get("url")
        content = item.get("content") or ""

        file_id = compute_file_id(title, publish_time)

        # 写入缓存（内存 + 本地文件）
        upsert_cache(
            file_id,
            title=title,
            publish_time=publish_time,
            url=url,
            content=content,
        )

        hit = ArticleHit(
            id=str(uuid.uuid4()),
            file_id=file_id,
            title=title,
            snippet=_make_snippet(content),
            score=1.0,
            publish_time=publish_time,
            url=url,
        )
        hits.append(hit)

    return hits

async def _search_articles(keyword: str, limit: int, search_engine="default") -> List[ArticleHit]:
    # weixin_search 内部是同步 requests；用线程池避免阻塞事件循环
    return await asyncio.to_thread(_search_articles_sync, keyword, limit, search_engine)

# ===================== 路由 =====================
@app.post("/api/search_keyword")
async def api_search_keyword(req: KeywordSearchRequest):
    """
    根据关键词搜索文章，或根据 file_ids 返回文章内容。
    请求体：
    {
      "keyword": "RAG",
      "limit": 5,        // 可选，默认10
      "file_ids": []     // 可选，若传则按id返回正文
    }

    响应（两种形态）：
    1) 普通搜索：，snippet是全文的部分内容
    {
      "keyword": "...",
      "count": 2,
      "mode": "search",
      "articles": [ {id, file_id, title, snippet, score, publish_time, url}, ... ]
    }

    2) 传 file_ids：content是完整内容
    {
      "keyword": "...",               // 原样返回
      "count": 2,
      "mode": "by_id",
      "articles": [ {file_id, title, content, publish_time, url}, ... ]
    }
    """
    keyword = req.keyword.strip()
    file_ids = req.file_ids or []

    if file_ids:
        docs: List[ArticleDoc] = []
        for fid in file_ids:
            cached = get_from_cache(fid)  # 先内存，后磁盘
            if not cached:
                # 没命中就跳过（也可记录 not_found_ids）
                continue
            docs.append(
                ArticleDoc(
                    file_id=fid,
                    title=cached.get("title", ""),
                    content=cached.get("content", ""),
                    publish_time=cached.get("publish_time"),
                    url=cached.get("url"),
                )
            )
        return JSONResponse(
            {
                "keyword": keyword,
                "count": len(docs),
                "mode": "by_id",
                "articles": [d.model_dump() for d in docs],
            }
        )

    # 正常搜索 -> 命中列表（并已落盘）
    hits = await _search_articles(keyword=keyword, limit=req.limit, search_engine=req.search_engine)
    return JSONResponse(
        {
            "keyword": keyword,
            "count": len(hits),
            "mode": "search",
            "articles": [h.model_dump() for h in hits],
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=10052)
