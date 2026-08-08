from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class CollectionCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    description: str | None = None


class Collection(BaseModel):
    id: UUID
    name: str
    description: str | None
    created_at: datetime


class DocumentStatus(BaseModel):
    id: UUID
    collection_id: UUID
    filename: str
    status: str
    error: str | None = None


class RetrievedChunk(BaseModel):
    """WS1が格納し、WS2の検索が返す、パイプライン共通のチャンク表現。"""

    chunk_id: UUID
    document_id: UUID
    filename: str
    heading_path: list[str] = []
    content: str
    score: float


class QueryRequest(BaseModel):
    collection_id: UUID
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=8, ge=1, le=20)
    stream: bool = True
    # falseで回答生成をスキップし出典のみ返す（検索精度評価用。生成クォータを消費しない）
    include_answer: bool = True


class Source(BaseModel):
    index: int
    chunk_id: UUID
    filename: str
    heading_path: list[str] = []
    content: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]
