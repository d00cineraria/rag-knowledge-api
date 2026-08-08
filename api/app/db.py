"""SQLiteデータベース層。

- sqlite-vec: ベクトル検索（vec0仮想テーブル、KNN）
- FTS5 (trigram): 日本語対応の全文検索（bm25()関数）
- WALモード: 取り込み(書き込み)とクエリ(読み取り)の並行を許容

単一ファイルDB（既定: ./data/rag.db）でローカル完結する。
"""

import json
from pathlib import Path

import aiosqlite
import sqlite_vec

from app.config import settings

_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS collections (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE,
    description TEXT,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS api_keys (
    id         TEXT PRIMARY KEY,
    key_hash   TEXT NOT NULL UNIQUE,
    label      TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS documents (
    id             TEXT PRIMARY KEY,
    collection_id  TEXT NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
    filename       TEXT NOT NULL,
    content_type   TEXT NOT NULL,
    content_sha256 TEXT NOT NULL,
    status         TEXT NOT NULL DEFAULT 'pending',
    error          TEXT,
    created_at     TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at     TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE (collection_id, content_sha256)
);

CREATE TABLE IF NOT EXISTS chunks (
    id            TEXT PRIMARY KEY,
    document_id   TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    collection_id TEXT NOT NULL,
    chunk_index   INTEGER NOT NULL,
    content       TEXT NOT NULL,
    heading_path  TEXT NOT NULL DEFAULT '[]',
    token_count   INTEGER,
    UNIQUE (document_id, chunk_index)
);
CREATE INDEX IF NOT EXISTS idx_chunks_collection ON chunks (collection_id);
"""

# FTS5(trigram)は日本語のようなスペース区切りのない言語でも部分文字列一致で検索できる
_SCHEMA_FTS = """
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    chunk_id UNINDEXED,
    content,
    tokenize='trigram'
);
"""

_conn: aiosqlite.Connection | None = None


def _schema_vec(dim: int) -> str:
    return (
        "CREATE VIRTUAL TABLE IF NOT EXISTS chunk_vectors USING vec0("
        f"chunk_id TEXT PRIMARY KEY, embedding float[{dim}]);"
    )


async def init_db() -> aiosqlite.Connection:
    global _conn
    if _conn is not None:
        return _conn
    db_path = Path(settings.sqlite_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = await aiosqlite.connect(db_path)
    await conn.enable_load_extension(True)
    await conn.load_extension(sqlite_vec.loadable_path())
    await conn.enable_load_extension(False)
    conn.row_factory = aiosqlite.Row
    await conn.executescript(_SCHEMA)
    await conn.executescript(_SCHEMA_FTS)
    await conn.executescript(_schema_vec(settings.embed_dim))
    await conn.commit()
    _conn = conn
    return conn


async def close_db() -> None:
    global _conn
    if _conn is not None:
        await _conn.close()
        _conn = None


def db() -> aiosqlite.Connection:
    if _conn is None:
        raise RuntimeError("DB not initialized")
    return _conn


def serialize_vector(vector: list[float]) -> bytes:
    """vec0のembedding列に渡すバイト列へ変換する。"""
    return sqlite_vec.serialize_float32(vector)


def dump_heading_path(heading_path: list[str]) -> str:
    return json.dumps(heading_path, ensure_ascii=False)


def load_heading_path(raw: str) -> list[str]:
    return json.loads(raw) if raw else []
