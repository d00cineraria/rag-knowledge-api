"""取り込みパイプラインの統合テスト。

一時ファイルの実SQLite（sqlite-vec + FTS5）に対して process_document を実行する。
外部I/OはGemini embeddingのみモックする。
"""

from uuid import uuid4

import pytest

from app import db as db_module
from app.config import settings
from app.services import ingest


class _FakeEmbedder:
    def __init__(self, dim: int = 768, fail: bool = False):
        self.dim = dim
        self.fail = fail

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if self.fail:
            raise RuntimeError("embedding failed (mock)")
        return [[1.0 / self.dim] * self.dim for _ in texts]


@pytest.fixture
async def sqlite_db(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "sqlite_path", str(tmp_path / "test.db"))
    monkeypatch.setattr(settings, "data_dir", str(tmp_path / "data"))
    conn = await db_module.init_db()
    yield conn
    await db_module.close_db()


async def _seed_document(conn, *, content: bytes, content_type: str = "text/markdown"):
    collection_id, document_id = str(uuid4()), str(uuid4())
    await conn.execute(
        "INSERT INTO collections (id, name) VALUES (?, ?)", (collection_id, f"c-{collection_id}")
    )
    await conn.execute(
        """
        INSERT INTO documents (id, collection_id, filename, content_type, content_sha256)
        VALUES (?, ?, 'doc.md', ?, ?)
        """,
        (document_id, collection_id, content_type, document_id),
    )
    await conn.commit()
    raw_dir = __import__("pathlib").Path(settings.data_dir) / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / document_id).write_bytes(content)
    return collection_id, document_id


async def _status(conn, document_id: str) -> tuple[str, str | None]:
    cursor = await conn.execute(
        "SELECT status, error FROM documents WHERE id = ?", (document_id,)
    )
    row = await cursor.fetchone()
    return row["status"], row["error"]


async def test_process_document_success_inserts_chunks_and_marks_ready(
    sqlite_db, monkeypatch
):
    monkeypatch.setattr(ingest, "GeminiEmbedder", _FakeEmbedder)
    _, document_id = await _seed_document(
        sqlite_db, content="# タイトル\n\n## 第1章\n\n本文です。\n".encode()
    )

    await ingest.process_document(document_id)

    status, error = await _status(sqlite_db, document_id)
    assert (status, error) == ("ready", None)
    cursor = await sqlite_db.execute(
        "SELECT COUNT(*) AS n FROM chunks WHERE document_id = ?", (document_id,)
    )
    n_chunks = (await cursor.fetchone())["n"]
    assert n_chunks >= 1
    cursor = await sqlite_db.execute("SELECT COUNT(*) AS n FROM chunks_fts")
    assert (await cursor.fetchone())["n"] == n_chunks
    cursor = await sqlite_db.execute("SELECT COUNT(*) AS n FROM chunk_vectors")
    assert (await cursor.fetchone())["n"] == n_chunks


async def test_process_document_missing_row_is_noop(sqlite_db):
    await ingest.process_document(str(uuid4()))  # 例外にならないこと


async def test_process_document_unsupported_content_type_sets_error(sqlite_db, monkeypatch):
    monkeypatch.setattr(ingest, "GeminiEmbedder", _FakeEmbedder)
    _, document_id = await _seed_document(
        sqlite_db, content=b"x", content_type="application/zip"
    )
    with pytest.raises(ValueError):
        await ingest.process_document(document_id)
    status, error = await _status(sqlite_db, document_id)
    assert status == "error"
    assert "unsupported" in (error or "")


async def test_process_document_empty_content_sets_error(sqlite_db, monkeypatch):
    monkeypatch.setattr(ingest, "GeminiEmbedder", _FakeEmbedder)
    _, document_id = await _seed_document(sqlite_db, content=b"")
    with pytest.raises(ValueError):
        await ingest.process_document(document_id)
    status, _ = await _status(sqlite_db, document_id)
    assert status == "error"


async def test_process_document_embedding_failure_sets_error(sqlite_db, monkeypatch):
    monkeypatch.setattr(
        ingest, "GeminiEmbedder", lambda: _FakeEmbedder(fail=True)
    )
    _, document_id = await _seed_document(sqlite_db, content="# t\n\n本文\n".encode())
    with pytest.raises(RuntimeError):
        await ingest.process_document(document_id)
    status, error = await _status(sqlite_db, document_id)
    assert status == "error"
    assert "mock" in (error or "")


async def test_process_document_worker_mode_stops_before_embedding(sqlite_db, monkeypatch):
    """INGEST_MODE=workerではGeminiEmbedderを呼ばず、chunks/chunks_ftsだけ書いて
    status='embedding'で止まる（chunk_vectorsは空のまま）。"""

    def _fail_if_constructed():
        raise AssertionError("worker mode must not construct GeminiEmbedder")

    monkeypatch.setattr(ingest, "GeminiEmbedder", lambda: _fail_if_constructed())
    monkeypatch.setattr(settings, "ingest_mode", "worker")

    _, document_id = await _seed_document(
        sqlite_db, content="# タイトル\n\n## 第1章\n\n本文です。\n".encode()
    )

    await ingest.process_document(document_id)

    status, error = await _status(sqlite_db, document_id)
    assert (status, error) == ("embedding", None)

    cursor = await sqlite_db.execute(
        "SELECT COUNT(*) AS n FROM chunks WHERE document_id = ?", (document_id,)
    )
    n_chunks = (await cursor.fetchone())["n"]
    assert n_chunks >= 1
    cursor = await sqlite_db.execute("SELECT COUNT(*) AS n FROM chunks_fts")
    assert (await cursor.fetchone())["n"] == n_chunks
    cursor = await sqlite_db.execute("SELECT COUNT(*) AS n FROM chunk_vectors")
    assert (await cursor.fetchone())["n"] == 0


async def test_process_document_inline_mode_unaffected_by_worker_mode_default(
    sqlite_db, monkeypatch
):
    """settings.ingest_modeの既定値'inline'では従来通りembeddingまで完結する。"""
    assert settings.ingest_mode == "inline"
    monkeypatch.setattr(ingest, "GeminiEmbedder", _FakeEmbedder)
    _, document_id = await _seed_document(
        sqlite_db, content="# タイトル\n\n## 第1章\n\n本文です。\n".encode()
    )

    await ingest.process_document(document_id)

    status, error = await _status(sqlite_db, document_id)
    assert (status, error) == ("ready", None)
    cursor = await sqlite_db.execute("SELECT COUNT(*) AS n FROM chunk_vectors")
    assert (await cursor.fetchone())["n"] >= 1


async def test_reingest_replaces_existing_chunks(sqlite_db, monkeypatch):
    monkeypatch.setattr(ingest, "GeminiEmbedder", _FakeEmbedder)
    _, document_id = await _seed_document(
        sqlite_db, content="# t\n\n## A\n\n本文A\n".encode()
    )
    await ingest.process_document(document_id)
    await ingest.process_document(document_id)  # 再取り込み

    cursor = await sqlite_db.execute(
        "SELECT COUNT(*) AS n FROM chunks WHERE document_id = ?", (document_id,)
    )
    n_chunks = (await cursor.fetchone())["n"]
    cursor = await sqlite_db.execute("SELECT COUNT(*) AS n FROM chunks_fts")
    assert (await cursor.fetchone())["n"] == n_chunks  # 二重登録されない
