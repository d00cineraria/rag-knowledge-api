"""Gemini embedding クライアントのラッパー。バッチ処理 + L2 正規化。"""

import math

from google import genai
from google.genai import types

from app.config import settings

_DEFAULT_BATCH_SIZE = 32


def _l2_normalize(values: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in values))
    if norm == 0.0:
        return values
    return [v / norm for v in values]


class GeminiEmbedder:
    """gemini-embedding-001 でテキストをバッチ埋め込みし、L2 正規化して返す。"""

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

    async def embed(self, texts: list[str]) -> list[list[float]]:
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
            results.extend(_l2_normalize(e.values or []) for e in embeddings)
        return results
