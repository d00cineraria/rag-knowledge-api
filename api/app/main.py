from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.auth import seed_dev_key
from app.config import settings
from app.db import close_db, init_db
from app.routers import collections, documents, query


@asynccontextmanager
async def lifespan(_: FastAPI):
    await init_db()
    await seed_dev_key(settings.api_dev_key)
    yield
    await close_db()


app = FastAPI(
    title="RAG Knowledge API",
    description="Markdown/PDFナレッジベースに出典引用付きでQAできるRAG API（評価基盤内蔵）",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(collections.router)
app.include_router(documents.router)
app.include_router(query.router)


@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok"}
