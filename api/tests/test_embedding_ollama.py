"""httpxをMockTransportで差し替えてOllamaEmbeddingProviderを検証する。

nomic-embed-textは非対称検索用にテキストへのプレフィックス（"search_document: " /
"search_query: "）が必要。実際のOllama疎通は行わない。
"""

import json
import math

import httpx
import pytest

from app.services.embedding import OllamaEmbeddingProvider


async def test_embed_documents_prefixes_and_normalizes():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        captured["input"] = body["input"]
        captured["model"] = body["model"]
        return httpx.Response(200, json={"embeddings": [[1.0, 2.0, 2.0] for _ in body["input"]]})

    provider = OllamaEmbeddingProvider(transport=httpx.MockTransport(handler))
    vectors = await provider.embed_documents(["本文A", "本文B"])

    assert captured["input"] == ["search_document: 本文A", "search_document: 本文B"]
    assert captured["model"] == provider._model
    for vec in vectors:
        assert math.sqrt(sum(v * v for v in vec)) == pytest.approx(1.0)


async def test_embed_documents_batches_by_batch_size():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append(body["input"])
        return httpx.Response(200, json={"embeddings": [[1.0, 0.0] for _ in body["input"]]})

    provider = OllamaEmbeddingProvider(batch_size=2, transport=httpx.MockTransport(handler))
    await provider.embed_documents(["a", "b", "c"])

    assert len(calls) == 2
    assert calls[0] == ["search_document: a", "search_document: b"]
    assert calls[1] == ["search_document: c"]


async def test_embed_documents_empty_list_returns_empty_without_request():
    called = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(200, json={"embeddings": []})

    provider = OllamaEmbeddingProvider(transport=httpx.MockTransport(handler))
    assert await provider.embed_documents([]) == []
    assert called is False


async def test_embed_query_prefixes_search_query():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        captured["input"] = body["input"]
        return httpx.Response(200, json={"embeddings": [[3.0, 4.0]]})

    provider = OllamaEmbeddingProvider(transport=httpx.MockTransport(handler))
    vector = await provider.embed_query("質問文")

    assert captured["input"] == ["search_query: 質問文"]
    assert vector == pytest.approx([0.6, 0.8])


async def test_embed_raises_on_response_size_mismatch():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"embeddings": [[1.0, 0.0]]})

    provider = OllamaEmbeddingProvider(transport=httpx.MockTransport(handler))
    with pytest.raises(RuntimeError):
        await provider.embed_documents(["a", "b"])
