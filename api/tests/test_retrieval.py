"""検索パイプラインのユニットテスト。DB・Gemini・CrossEncoderは全てモック。"""

from uuid import uuid4

import pytest

from app.schemas import RetrievedChunk
from app.services import retrieval


def test_reciprocal_rank_fusion_sums_inverse_rank():
    a, b, c = uuid4(), uuid4(), uuid4()
    scores = retrieval.reciprocal_rank_fusion([[a, b], [b, c]], k=60)
    assert scores[a] == pytest.approx(1 / 61)
    assert scores[b] == pytest.approx(1 / 62 + 1 / 61)
    assert scores[c] == pytest.approx(1 / 62)


def test_reciprocal_rank_fusion_ranks_items_in_both_lists_higher():
    a, b, c = uuid4(), uuid4(), uuid4()
    scores = retrieval.reciprocal_rank_fusion([[a, b, c], [c, a, b]], k=60)
    ordered = sorted(scores, key=lambda cid: scores[cid], reverse=True)
    assert set(ordered[:2]) == {a, c}
    assert ordered[2] == b


def test_reciprocal_rank_fusion_empty_lists_returns_empty():
    assert retrieval.reciprocal_rank_fusion([[], []]) == {}


def test_normalize_scales_to_unit_length():
    vec = retrieval.normalize([3.0, 4.0])
    assert vec == pytest.approx([0.6, 0.8])


def test_normalize_zero_vector_is_unchanged():
    assert retrieval.normalize([0.0, 0.0]) == [0.0, 0.0]


@pytest.fixture(autouse=True)
def _reranker_disabled_by_default(monkeypatch):
    monkeypatch.setattr(retrieval.settings, "reranker_enabled", False)


def _make_chunk(chunk_id, content="本文"):
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=uuid4(),
        filename="doc.md",
        heading_path=["第1章"],
        content=content,
        score=0.0,
    )


async def test_search_fuses_bm25_and_vector_then_returns_top_k(monkeypatch):
    bm25_id, vec_id, both_id = uuid4(), uuid4(), uuid4()
    chunks = {
        bm25_id: _make_chunk(bm25_id),
        vec_id: _make_chunk(vec_id),
        both_id: _make_chunk(both_id),
    }

    async def fake_embed(question):
        return [0.1, 0.2]

    async def fake_bm25(collection_id, question, limit):
        return [both_id, bm25_id]

    async def fake_vector(collection_id, qvec, limit):
        assert qvec == [0.1, 0.2]
        return [both_id, vec_id]

    async def fake_fetch(chunk_ids):
        return {cid: chunks[cid] for cid in chunk_ids}

    monkeypatch.setattr(retrieval, "_embed_question", fake_embed)
    monkeypatch.setattr(retrieval, "_bm25_candidate_ids", fake_bm25)
    monkeypatch.setattr(retrieval, "_vector_candidate_ids", fake_vector)
    monkeypatch.setattr(retrieval, "_fetch_chunks_by_id", fake_fetch)

    result = await retrieval.search(uuid4(), "質問", top_k=2)

    assert [c.chunk_id for c in result] == [both_id, bm25_id]
    assert result[0].score == pytest.approx(2 / 61)
    assert result[1].score == pytest.approx(1 / 62)


async def test_search_skips_candidates_missing_from_documents_join(monkeypatch):
    kept_id, missing_id = uuid4(), uuid4()
    chunks = {kept_id: _make_chunk(kept_id)}

    async def fake_embed(question):
        return [0.0]

    async def fake_bm25(collection_id, question, limit):
        return [missing_id, kept_id]

    async def fake_vector(collection_id, qvec, limit):
        return []

    async def fake_fetch(chunk_ids):
        return {cid: chunks[cid] for cid in chunk_ids if cid in chunks}

    monkeypatch.setattr(retrieval, "_embed_question", fake_embed)
    monkeypatch.setattr(retrieval, "_bm25_candidate_ids", fake_bm25)
    monkeypatch.setattr(retrieval, "_vector_candidate_ids", fake_vector)
    monkeypatch.setattr(retrieval, "_fetch_chunks_by_id", fake_fetch)

    result = await retrieval.search(uuid4(), "質問", top_k=8)

    assert [c.chunk_id for c in result] == [kept_id]


async def test_search_applies_reranker_when_enabled(monkeypatch):
    monkeypatch.setattr(retrieval.settings, "reranker_enabled", True)
    a, b = uuid4(), uuid4()
    chunks = {a: _make_chunk(a, content="低スコア"), b: _make_chunk(b, content="高スコア")}

    async def fake_embed(question):
        return [0.0]

    async def fake_bm25(collection_id, question, limit):
        return [a, b]

    async def fake_vector(collection_id, qvec, limit):
        return []

    async def fake_fetch(chunk_ids):
        return {cid: chunks[cid] for cid in chunk_ids}

    class FakeReranker:
        def predict(self, pairs):
            return [0.1 if content == "低スコア" else 0.9 for _, content in pairs]

    monkeypatch.setattr(retrieval, "_embed_question", fake_embed)
    monkeypatch.setattr(retrieval, "_bm25_candidate_ids", fake_bm25)
    monkeypatch.setattr(retrieval, "_vector_candidate_ids", fake_vector)
    monkeypatch.setattr(retrieval, "_fetch_chunks_by_id", fake_fetch)
    monkeypatch.setattr(retrieval, "_load_reranker", lambda: FakeReranker())

    result = await retrieval.search(uuid4(), "質問", top_k=2)

    assert [c.chunk_id for c in result] == [b, a]
    assert result[0].score == pytest.approx(0.9)
    assert result[1].score == pytest.approx(0.1)
