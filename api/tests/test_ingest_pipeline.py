"""GeminiクライアントとDBプールをモックしてprocess_documentの制御フローを検証する。"""

from pathlib import Path
from uuid import uuid4

import pytest

import app.services.ingest as ingest_module
from app.config import settings


class _NullTransaction:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakeConn:
    def __init__(self):
        self.deleted: list[tuple] = []
        self.inserted: list[tuple] = []

    async def execute(self, query, *args):
        assert query.strip().startswith("DELETE")
        self.deleted.append(args)

    async def executemany(self, query, args_list):
        self.inserted.extend(args_list)

    def transaction(self):
        return _NullTransaction()


class _AcquireContext:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *exc):
        return False


class _FakePool:
    def __init__(self, row):
        self._row = row
        self.conn = _FakeConn()
        self.updates: list[tuple[str, tuple]] = []

    async def fetchrow(self, query, *args):
        return self._row

    async def execute(self, query, *args):
        self.updates.append((query, args))

    def acquire(self):
        return _AcquireContext(self.conn)


class _FakeEmbedder:
    def __init__(self, *args, **kwargs):
        pass

    async def embed(self, texts):
        return [[1.0, 0.0, 0.0] for _ in texts]


def _status_updates(pool: _FakePool, status: str) -> list[tuple[str, tuple]]:
    return [(q, a) for q, a in pool.updates if f"'{status}'" in q]


@pytest.fixture(autouse=True)
def _isolate_data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", str(tmp_path))


def _write_raw(document_id, content: bytes) -> None:
    raw_dir = Path(settings.data_dir) / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / str(document_id)).write_bytes(content)


async def test_process_document_success_inserts_chunks_and_marks_ready(monkeypatch):
    document_id = uuid4()
    collection_id = uuid4()
    _write_raw(document_id, "# 見出し\n\n本文です。\n".encode())

    fake_pool = _FakePool({"collection_id": collection_id, "content_type": "text/markdown"})
    monkeypatch.setattr(ingest_module, "pool", lambda: fake_pool)
    monkeypatch.setattr(ingest_module, "GeminiEmbedder", _FakeEmbedder)

    await ingest_module.process_document(document_id)

    assert len(_status_updates(fake_pool, "processing")) == 1
    ready_updates = _status_updates(fake_pool, "ready")
    assert len(ready_updates) == 1
    assert ready_updates[0][1][0] == document_id
    assert _status_updates(fake_pool, "error") == []

    assert len(fake_pool.conn.inserted) == 1
    row = fake_pool.conn.inserted[0]
    doc_id, coll_id, chunk_index, content, heading_path, token_count, embedding = row
    assert doc_id == document_id
    assert coll_id == collection_id
    assert chunk_index == 0
    assert content == "本文です。"
    assert heading_path == ["見出し"]
    assert token_count >= 1
    assert embedding == [1.0, 0.0, 0.0]


async def test_process_document_missing_row_is_noop(monkeypatch):
    fake_pool = _FakePool(None)
    monkeypatch.setattr(ingest_module, "pool", lambda: fake_pool)

    await ingest_module.process_document(uuid4())

    assert fake_pool.updates == []


async def test_process_document_unsupported_content_type_sets_error(monkeypatch):
    document_id = uuid4()
    collection_id = uuid4()
    _write_raw(document_id, b"dummy")

    fake_pool = _FakePool(
        {"collection_id": collection_id, "content_type": "application/octet-stream"}
    )
    monkeypatch.setattr(ingest_module, "pool", lambda: fake_pool)
    monkeypatch.setattr(ingest_module, "GeminiEmbedder", _FakeEmbedder)

    with pytest.raises(ValueError, match="unsupported content_type"):
        await ingest_module.process_document(document_id)

    error_updates = _status_updates(fake_pool, "error")
    assert len(error_updates) == 1
    assert "unsupported content_type" in error_updates[0][1][1]
    assert fake_pool.conn.inserted == []


async def test_process_document_empty_content_sets_error(monkeypatch):
    document_id = uuid4()
    collection_id = uuid4()
    _write_raw(document_id, b"   \n\n  ")

    fake_pool = _FakePool({"collection_id": collection_id, "content_type": "text/markdown"})
    monkeypatch.setattr(ingest_module, "pool", lambda: fake_pool)
    monkeypatch.setattr(ingest_module, "GeminiEmbedder", _FakeEmbedder)

    with pytest.raises(ValueError, match="no extractable content"):
        await ingest_module.process_document(document_id)

    error_updates = _status_updates(fake_pool, "error")
    assert len(error_updates) == 1


async def test_process_document_embedding_failure_sets_error(monkeypatch):
    document_id = uuid4()
    collection_id = uuid4()
    _write_raw(document_id, "# 見出し\n\n本文です。\n".encode())

    fake_pool = _FakePool({"collection_id": collection_id, "content_type": "text/markdown"})
    monkeypatch.setattr(ingest_module, "pool", lambda: fake_pool)

    class _FailingEmbedder:
        def __init__(self, *args, **kwargs):
            pass

        async def embed(self, texts):
            raise RuntimeError("gemini api down")

    monkeypatch.setattr(ingest_module, "GeminiEmbedder", _FailingEmbedder)

    with pytest.raises(RuntimeError, match="gemini api down"):
        await ingest_module.process_document(document_id)

    error_updates = _status_updates(fake_pool, "error")
    assert len(error_updates) == 1
    assert "gemini api down" in error_updates[0][1][1]
    assert fake_pool.conn.inserted == []
