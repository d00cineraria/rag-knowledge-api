"""ハイブリッド検索（WS2が実装）。

契約: docs/contracts.md 参照。
PGroonga BM25 + pgvector cosine を各30件→RRF(k=60)融合→
RERANKER_ENABLED時は bge-reranker-v2-m3 でリランク→top_k件。
"""

import asyncio
from uuid import UUID

from google import genai
from google.genai import types as genai_types

from app.config import settings
from app.db import pool
from app.schemas import RetrievedChunk

CANDIDATE_LIMIT = 30
RRF_K = 60

_gemini_client: genai.Client | None = None
_reranker = None


def reciprocal_rank_fusion(
    ranked_id_lists: list[list[UUID]], k: int = RRF_K
) -> dict[UUID, float]:
    """RRF(k)融合: score = Σ 1/(k+rank)（rankは1始まり）。

    複数の順位リストを1つのスコアに統合する純粋関数。
    """
    scores: dict[UUID, float] = {}
    for ranked_ids in ranked_id_lists:
        for rank, chunk_id in enumerate(ranked_ids, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return scores


def normalize(vector: list[float]) -> list[float]:
    """L2正規化。ingestと同条件でGemini embeddingを正規化する。"""
    norm = sum(v * v for v in vector) ** 0.5
    if norm == 0.0:
        return vector
    return [v / norm for v in vector]


def _get_gemini_client() -> genai.Client:
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = genai.Client(api_key=settings.gemini_api_key)
    return _gemini_client


async def _embed_question(question: str) -> list[float]:
    client = _get_gemini_client()
    response = await client.aio.models.embed_content(
        model=settings.gemini_embed_model,
        contents=question,
        config=genai_types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=settings.embed_dim,
        ),
    )
    return normalize(response.embeddings[0].values)


async def _bm25_candidate_ids(collection_id: UUID, question: str, limit: int) -> list[UUID]:
    rows = await pool().fetch(
        """
        SELECT id
        FROM chunks
        WHERE collection_id = $1 AND content &@~ $2
        ORDER BY pgroonga_score(tableoid, ctid) DESC
        LIMIT $3
        """,
        collection_id,
        question,
        limit,
    )
    return [row["id"] for row in rows]


async def _vector_candidate_ids(collection_id: UUID, qvec: list[float], limit: int) -> list[UUID]:
    rows = await pool().fetch(
        """
        SELECT id
        FROM chunks
        WHERE collection_id = $1
        ORDER BY embedding <=> $2
        LIMIT $3
        """,
        collection_id,
        qvec,
        limit,
    )
    return [row["id"] for row in rows]


async def _fetch_chunks_by_id(chunk_ids: list[UUID]) -> dict[UUID, RetrievedChunk]:
    if not chunk_ids:
        return {}
    rows = await pool().fetch(
        """
        SELECT c.id AS chunk_id, c.document_id, d.filename, c.heading_path, c.content
        FROM chunks c
        JOIN documents d ON d.id = c.document_id
        WHERE c.id = ANY($1::uuid[])
        """,
        chunk_ids,
    )
    return {
        row["chunk_id"]: RetrievedChunk(
            chunk_id=row["chunk_id"],
            document_id=row["document_id"],
            filename=row["filename"],
            heading_path=list(row["heading_path"]),
            content=row["content"],
            score=0.0,
        )
        for row in rows
    }


def _load_reranker():
    """bge-reranker-v2-m3の遅延ロード・シングルトン（CPU）。RERANKER_ENABLED時のみimportされる。"""
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder

        _reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", device="cpu")
    return _reranker


async def _rerank(
    question: str, chunks: list[RetrievedChunk], top_k: int
) -> list[RetrievedChunk]:
    reranker = await asyncio.to_thread(_load_reranker)
    pairs = [(question, chunk.content) for chunk in chunks]
    scores = await asyncio.to_thread(reranker.predict, pairs)
    reranked = sorted(zip(chunks, scores, strict=True), key=lambda pair: pair[1], reverse=True)
    return [
        chunk.model_copy(update={"score": float(score)}) for chunk, score in reranked[:top_k]
    ]


async def search(collection_id: UUID, question: str, top_k: int = 8) -> list[RetrievedChunk]:
    qvec = await _embed_question(question)
    bm25_ids, vector_ids = await asyncio.gather(
        _bm25_candidate_ids(collection_id, question, CANDIDATE_LIMIT),
        _vector_candidate_ids(collection_id, qvec, CANDIDATE_LIMIT),
    )

    fused_scores = reciprocal_rank_fusion([bm25_ids, vector_ids])
    ordered_ids = sorted(fused_scores, key=lambda cid: fused_scores[cid], reverse=True)

    fetch_count = CANDIDATE_LIMIT if settings.reranker_enabled else top_k
    candidate_ids = ordered_ids[:fetch_count]

    chunks_by_id = await _fetch_chunks_by_id(candidate_ids)
    ranked_chunks = [
        chunks_by_id[cid].model_copy(update={"score": fused_scores[cid]})
        for cid in candidate_ids
        if cid in chunks_by_id
    ]

    if settings.reranker_enabled and ranked_chunks:
        return await _rerank(question, ranked_chunks, top_k)

    return ranked_chunks[:top_k]
