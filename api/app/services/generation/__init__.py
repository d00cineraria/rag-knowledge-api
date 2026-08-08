"""回答生成（WS2が実装）。

契約: docs/contracts.md 参照。
Geminiで出典引用付き回答をトークン単位でyield。出典は[1][2]形式。
"""

from collections.abc import AsyncIterator

from app.schemas import RetrievedChunk


async def stream_answer(question: str, chunks: list[RetrievedChunk]) -> AsyncIterator[str]:
    raise NotImplementedError("WS2: generation not implemented yet")
    yield  # pragma: no cover
