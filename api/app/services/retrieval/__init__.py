"""ハイブリッド検索（WS2が実装）。

契約: docs/contracts.md 参照。
PGroonga BM25 + pgvector cosine を各30件→RRF(k=60)融合→
RERANKER_ENABLED時は bge-reranker-v2-m3 でリランク→top_k件。
"""

from uuid import UUID

from app.schemas import RetrievedChunk


async def search(collection_id: UUID, question: str, top_k: int = 8) -> list[RetrievedChunk]:
    raise NotImplementedError("WS2: retrieval not implemented yet")
