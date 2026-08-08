"""GeminiクライアントをモックしてGeminiEmbeddingProviderのバッチ処理・L2正規化・
質問埋め込み(RETRIEVAL_QUERY)を検証する。"""

import math

import pytest

from app.services.embedding import GeminiEmbeddingProvider


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
        self._calls.append((contents, config))
        if isinstance(contents, str):
            return _FakeResponse([_FakeEmbedding(list(self._values))])
        return _FakeResponse([_FakeEmbedding(list(self._values)) for _ in contents])


class _FakeAio:
    def __init__(self, calls, **kwargs):
        self.models = _FakeModels(calls, **kwargs)


class _FakeClient:
    def __init__(self, calls, **kwargs):
        self.aio = _FakeAio(calls, **kwargs)


async def test_embed_documents_l2_normalizes_and_batches():
    calls: list = []
    provider = GeminiEmbeddingProvider(client=_FakeClient(calls), batch_size=2)

    vectors = await provider.embed_documents(["a", "b", "c"])

    assert len(vectors) == 3
    for vec in vectors:
        norm = math.sqrt(sum(v * v for v in vec))
        assert norm == pytest.approx(1.0)
    assert [contents for contents, _ in calls] == [["a", "b"], ["c"]]


async def test_embed_documents_empty_list_returns_empty_without_api_call():
    calls: list = []
    provider = GeminiEmbeddingProvider(client=_FakeClient(calls))

    assert await provider.embed_documents([]) == []
    assert calls == []


async def test_embed_documents_raises_on_response_size_mismatch():
    class _MismatchModels:
        async def embed_content(self, **kwargs):
            return _FakeResponse([_FakeEmbedding([1.0, 0.0])])

    class _MismatchAio:
        models = _MismatchModels()

    class _MismatchClient:
        aio = _MismatchAio()

    provider = GeminiEmbeddingProvider(client=_MismatchClient())

    with pytest.raises(RuntimeError):
        await provider.embed_documents(["a", "b"])


async def test_embed_query_uses_retrieval_query_task_type_and_normalizes():
    calls: list = []
    provider = GeminiEmbeddingProvider(client=_FakeClient(calls))

    vector = await provider.embed_query("質問")

    norm = math.sqrt(sum(v * v for v in vector))
    assert norm == pytest.approx(1.0)
    [(contents, config)] = calls
    assert contents == "質問"
    assert config.task_type == "RETRIEVAL_QUERY"
