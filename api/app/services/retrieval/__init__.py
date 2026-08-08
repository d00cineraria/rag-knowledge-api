"""ハイブリッド検索。

FTS5(trigram) BM25 + sqlite-vec cosine を各30件→RRF(k=60)融合→
RERANKER_ENABLED時は bge-reranker-v2-m3 でリランク→top_k件。
"""

import asyncio
import re
from uuid import UUID

from google import genai
from google.genai import types as genai_types

from app.config import settings
from app.db import db, load_heading_path, serialize_vector
from app.schemas import RetrievedChunk

CANDIDATE_LIMIT = 30
RRF_K = 60

# 日本語の質問文をFTS5(trigram)のOR検索に分解するための区切り
# （助詞・助動詞・句読点・記号で分割し、2文字以上の断片をキーワードとして使う）
_SPLIT_PATTERN = re.compile(
    r"[はがをにでとへもやの]|から|まで|より|など|について|ですか|ますか|でしょうか"
    r"|です|ます|する|されて|して|といった"
    r"|[\s、。・？！?!「」『』（）()\[\]{}:：;；,，.]+"
)

_gemini_client: genai.Client | None = None
_reranker = None


def build_fts_query(question: str) -> str:
    """質問文をFTS5のOR検索クエリへ変換する純粋関数。

    trigramトークナイザは3文字以上の部分文字列一致で検索するため、
    助詞等で分割した2文字以上の断片をダブルクォートで包みORで結合する。
    """
    fragments = [f for f in _SPLIT_PATTERN.split(question) if f and len(f) >= 2]
    seen: dict[str, None] = {}
    for fragment in fragments:
        seen.setdefault(fragment.replace('"', ""), None)
    if not seen:
        return f'"{question.replace(chr(34), "")}"'
    return " OR ".join(f'"{fragment}"' for fragment in seen)


def reciprocal_rank_fusion(
    ranked_id_lists: list[list[str]], k: int = RRF_K
) -> dict[str, float]:
    """RRF(k)融合: score = Σ 1/(k+rank)（rankは1始まり）。"""
    scores: dict[str, float] = {}
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


async def _bm25_candidate_ids(collection_id: str, question: str, limit: int) -> list[str]:
    """FTS5のbm25()昇順（小さいほど良い）で候補chunk_idを返す。"""
    fts_query = build_fts_query(question)
    cursor = await db().execute(
        """
        SELECT f.chunk_id
        FROM chunks_fts f
        JOIN chunks c ON c.id = f.chunk_id
        WHERE chunks_fts MATCH ? AND c.collection_id = ?
        ORDER BY bm25(chunks_fts)
        LIMIT ?
        """,
        (fts_query, collection_id, limit),
    )
    return [row["chunk_id"] for row in await cursor.fetchall()]


async def _vector_candidate_ids(collection_id: str, qvec: list[float], limit: int) -> list[str]:
    """sqlite-vecのKNN。コレクション横断で多めに取り、collection_idで絞り込む。"""
    cursor = await db().execute(
        """
        SELECT v.chunk_id, v.distance
        FROM chunk_vectors v
        WHERE v.embedding MATCH ? AND k = ?
        ORDER BY v.distance
        """,
        (serialize_vector(qvec), limit * 4),
    )
    rows = await cursor.fetchall()
    if not rows:
        return []
    candidate_ids = [row["chunk_id"] for row in rows]
    placeholders = ",".join("?" for _ in candidate_ids)
    cursor = await db().execute(
        f"SELECT id FROM chunks WHERE id IN ({placeholders}) AND collection_id = ?",
        (*candidate_ids, collection_id),
    )
    in_collection = {row["id"] for row in await cursor.fetchall()}
    return [cid for cid in candidate_ids if cid in in_collection][:limit]


async def _fetch_chunks_by_id(chunk_ids: list[str]) -> dict[str, RetrievedChunk]:
    if not chunk_ids:
        return {}
    placeholders = ",".join("?" for _ in chunk_ids)
    cursor = await db().execute(
        f"""
        SELECT c.id AS chunk_id, c.document_id, d.filename, c.heading_path, c.content
        FROM chunks c
        JOIN documents d ON d.id = c.document_id
        WHERE c.id IN ({placeholders})
        """,
        chunk_ids,
    )
    rows = await cursor.fetchall()
    return {
        row["chunk_id"]: RetrievedChunk(
            chunk_id=row["chunk_id"],
            document_id=row["document_id"],
            filename=row["filename"],
            heading_path=load_heading_path(row["heading_path"]),
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
    cid = str(collection_id)
    bm25_ids = await _bm25_candidate_ids(cid, question, CANDIDATE_LIMIT)
    vector_ids = await _vector_candidate_ids(cid, qvec, CANDIDATE_LIMIT)

    fused_scores = reciprocal_rank_fusion([bm25_ids, vector_ids])
    ordered_ids = sorted(fused_scores, key=lambda chunk_id: fused_scores[chunk_id], reverse=True)

    fetch_count = CANDIDATE_LIMIT if settings.reranker_enabled else top_k
    candidate_ids = ordered_ids[:fetch_count]

    chunks_by_id = await _fetch_chunks_by_id(candidate_ids)
    ranked_chunks = [
        chunks_by_id[cid_].model_copy(update={"score": fused_scores[cid_]})
        for cid_ in candidate_ids
        if cid_ in chunks_by_id
    ]

    if settings.reranker_enabled and ranked_chunks:
        return await _rerank(question, ranked_chunks, top_k)

    return ranked_chunks[:top_k]
