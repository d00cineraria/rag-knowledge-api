"""embeddingプロバイダの抽象化。

ingest（バッチ埋め込み）とretrieval（質問埋め込み）が重複して直接Gemini APIを
叩いていたのを1つの抽象へ統合する。settings.llm_provider に応じて
GeminiEmbeddingProvider / OllamaEmbeddingProvider を返す。
"""

from typing import Protocol

import httpx
from google import genai
from google.genai import types

from app.config import settings

from .normalize import l2_normalize

_DEFAULT_BATCH_SIZE = 32


class EmbeddingProvider(Protocol):
    async def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    async def embed_query(self, text: str) -> list[float]: ...


class GeminiEmbeddingProvider:
    """gemini-embedding-001でテキストをバッチ埋め込みし、L2正規化して返す。"""

    def __init__(
        self,
        client: genai.Client | None = None,
        model: str = settings.gemini_embed_model,
        output_dimensionality: int = settings.embed_dim,
        batch_size: int = _DEFAULT_BATCH_SIZE,
    ) -> None:
        self._client = client or genai.Client(api_key=settings.gemini_api_key)
        self._model = model
        self._output_dimensionality = output_dimensionality
        self._batch_size = batch_size

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        results: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            response = await self._client.aio.models.embed_content(
                model=self._model,
                contents=batch,
                config=types.EmbedContentConfig(
                    output_dimensionality=self._output_dimensionality
                ),
            )
            embeddings = response.embeddings or []
            if len(embeddings) != len(batch):
                raise RuntimeError("Gemini embedding response size mismatch")
            results.extend(l2_normalize(e.values or []) for e in embeddings)
        return results

    async def embed_query(self, text: str) -> list[float]:
        response = await self._client.aio.models.embed_content(
            model=self._model,
            contents=text,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY",
                output_dimensionality=self._output_dimensionality,
            ),
        )
        return l2_normalize(response.embeddings[0].values)


class OllamaEmbeddingProvider:
    """nomic-embed-textでテキストを埋め込む。

    非対称検索用のプレフィックス（"search_document: " / "search_query: "）は
    Ollama側のAPIパラメータではなくテキスト自体への前置が必要。
    """

    def __init__(
        self,
        base_url: str = settings.ollama_base_url,
        model: str = settings.ollama_embed_model,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._base_url = base_url
        self._model = model
        self._batch_size = batch_size
        self._transport = transport

    async def _embed(self, inputs: list[str]) -> list[list[float]]:
        async with httpx.AsyncClient(
            base_url=self._base_url, timeout=120.0, transport=self._transport
        ) as client:
            response = await client.post("/api/embed", json={"model": self._model, "input": inputs})
            response.raise_for_status()
            data = response.json()
        embeddings = data.get("embeddings") or []
        if len(embeddings) != len(inputs):
            raise RuntimeError("Ollama embedding response size mismatch")
        return [l2_normalize(vec) for vec in embeddings]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        results: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            results.extend(await self._embed([f"search_document: {t}" for t in batch]))
        return results

    async def embed_query(self, text: str) -> list[float]:
        vectors = await self._embed([f"search_query: {text}"])
        return vectors[0]


def get_embedding_provider() -> EmbeddingProvider:
    """settings.llm_provider に応じてGemini/Ollama実装を返す。"""
    if settings.llm_provider == "gemini":
        return GeminiEmbeddingProvider()
    return OllamaEmbeddingProvider()
