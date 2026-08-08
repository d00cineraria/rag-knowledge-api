"""実SQLite（sqlite-vec + FTS5 trigram）に対するハイブリッド検索の統合テスト。

外部I/Oはembeddingプロバイダ（質問ベクトル化）のみモックする。
"""

from uuid import uuid4

import pytest

from app import db as db_module
from app.config import settings
from app.db import dump_heading_path, serialize_vector
from app.services import retrieval


@pytest.fixture
async def sqlite_db(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "sqlite_path", str(tmp_path / "test.db"))
    conn = await db_module.init_db()
    yield conn
    await db_module.close_db()


def test_build_fts_query_splits_japanese_particles():
    q = retrieval.build_fts_query("有給休暇の繰越はどうなっていますか")
    assert '"有給休暇"' in q
    assert '"繰越"' in q
    assert " OR " in q


def test_build_fts_query_never_returns_empty():
    assert retrieval.build_fts_query("の") != ""


async def _seed_chunk(conn, collection_id: str, content: str, vector: list[float]) -> str:
    document_id, chunk_id = str(uuid4()), str(uuid4())
    await conn.execute(
        "INSERT OR IGNORE INTO collections (id, name) VALUES (?, ?)",
        (collection_id, f"c-{collection_id}"),
    )
    await conn.execute(
        """
        INSERT INTO documents (id, collection_id, filename, content_type, content_sha256, status)
        VALUES (?, ?, 'doc.md', 'text/markdown', ?, 'ready')
        """,
        (document_id, collection_id, document_id),
    )
    await conn.execute(
        """
        INSERT INTO chunks (id, document_id, collection_id, chunk_index, content, heading_path)
        VALUES (?, ?, ?, 0, ?, ?)
        """,
        (chunk_id, document_id, collection_id, content, dump_heading_path(["第1章"])),
    )
    await conn.execute(
        "INSERT INTO chunks_fts (chunk_id, content) VALUES (?, ?)", (chunk_id, content)
    )
    await conn.execute(
        "INSERT INTO chunk_vectors (chunk_id, embedding) VALUES (?, ?)",
        (chunk_id, serialize_vector(vector)),
    )
    await conn.commit()
    return chunk_id


async def test_search_returns_bm25_and_vector_hits_from_real_db(sqlite_db, monkeypatch):
    monkeypatch.setattr(retrieval.settings, "reranker_enabled", False)
    dim = settings.embed_dim
    collection_id = str(uuid4())

    vec_a = [1.0] + [0.0] * (dim - 1)
    vec_b = [0.0, 1.0] + [0.0] * (dim - 2)
    target = await _seed_chunk(
        sqlite_db, collection_id, "年次有給休暇の繰越日数の上限は20日です。", vec_a
    )
    await _seed_chunk(sqlite_db, collection_id, "経費精算は月末締めで翌月払いです。", vec_b)

    async def fake_embed(question):
        return [1.0] + [0.0] * (dim - 1)  # targetのベクトルに一致

    monkeypatch.setattr(retrieval, "_embed_question", fake_embed)

    result = await retrieval.search(collection_id, "有給休暇の繰越の上限は？", top_k=2)

    assert result
    assert str(result[0].chunk_id) == target  # BM25・ベクトル双方で1位 → RRF融合でも1位
    assert result[0].heading_path == ["第1章"]
    assert result[0].filename == "doc.md"


async def test_search_excludes_other_collections(sqlite_db, monkeypatch):
    monkeypatch.setattr(retrieval.settings, "reranker_enabled", False)
    dim = settings.embed_dim
    mine, other = str(uuid4()), str(uuid4())

    vec = [1.0] + [0.0] * (dim - 1)
    await _seed_chunk(sqlite_db, mine, "リモートワークは週3日まで可能です。", vec)
    await _seed_chunk(sqlite_db, other, "リモートワークは全面禁止です。", vec)

    async def fake_embed(question):
        return [1.0] + [0.0] * (dim - 1)

    monkeypatch.setattr(retrieval, "_embed_question", fake_embed)

    result = await retrieval.search(mine, "リモートワークは何日まで？", top_k=8)

    assert result
    assert all("禁止" not in chunk.content for chunk in result)
