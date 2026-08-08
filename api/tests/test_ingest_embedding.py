"""GeminiクライアントをモックしてGeminiEmbedderのバッチ処理とL2正規化を検証する。"""

import math

import pytest

from app.services.ingest.embedding import GeminiEmbedder


class _FakeEmbedding:
    def __init__(self, values):
        self.values = values


class _FakeResponse:
    def __init__(self, embeddings):
        self.embeddings = embeddings


class _FakeModels:
    def __init__(self, calls, values_per_text=(1.0, 2.0, 2.0)):
        self._calls = calls
        self._values = values_per_text

    async def embed_content(self, *, model, contents, config):
        self._calls.append(list(contents))
        return _FakeResponse([_FakeEmbedding(list(self._values)) for _ in contents])


class _FakeAio:
    def __init__(self, calls, **kwargs):
        self.models = _FakeModels(calls, **kwargs)


class _FakeClient:
    def __init__(self, calls, **kwargs):
        self.aio = _FakeAio(calls, **kwargs)


async def test_embed_l2_normalizes_and_batches():
    calls: list[list[str]] = []
    embedder = GeminiEmbedder(client=_FakeClient(calls), batch_size=2)

    vectors = await embedder.embed(["a", "b", "c"])

    assert len(vectors) == 3
    for vec in vectors:
        norm = math.sqrt(sum(v * v for v in vec))
        assert norm == pytest.approx(1.0)
    assert calls == [["a", "b"], ["c"]]


async def test_embed_empty_list_returns_empty_without_api_call():
    calls: list[list[str]] = []
    embedder = GeminiEmbedder(client=_FakeClient(calls))

    assert await embedder.embed([]) == []
    assert calls == []


async def test_embed_raises_on_response_size_mismatch():
    class _MismatchModels:
        async def embed_content(self, **kwargs):
            return _FakeResponse([_FakeEmbedding([1.0, 0.0])])

    class _MismatchAio:
        models = _MismatchModels()

    class _MismatchClient:
        aio = _MismatchAio()

    embedder = GeminiEmbedder(client=_MismatchClient())

    with pytest.raises(RuntimeError):
        await embedder.embed(["a", "b"])
