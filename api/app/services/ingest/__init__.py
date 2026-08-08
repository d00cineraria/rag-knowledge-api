"""取り込みパイプライン。

./data/raw/{document_id} のファイルをパース→見出し考慮チャンキング→
embedding（768次元・正規化、Gemini/OllamaはEMBEDDING_PROVIDERで切替）→
chunks / chunks_fts / chunk_vectors へ INSERT → documents.status を更新する。
"""

import logging
from pathlib import Path
from uuid import UUID, uuid4

from app.config import settings
from app.db import db, dump_heading_path, serialize_vector
from app.services.embedding import get_embedding_provider

from .chunking import Chunk, chunk_markdown, chunk_pdf_pages
from .parsing import extract_pdf_pages

logger = logging.getLogger(__name__)

_MARKDOWN = "text/markdown"
_PDF = "application/pdf"


def _estimate_token_count(text: str) -> int:
    return max(1, len(text) // 4)


def _parse_chunks(content_type: str, data: bytes) -> list[Chunk]:
    if content_type == _MARKDOWN:
        return chunk_markdown(data.decode("utf-8"))
    if content_type == _PDF:
        return chunk_pdf_pages(extract_pdf_pages(data))
    raise ValueError(f"unsupported content_type: {content_type}")


async def process_document(document_id: UUID) -> None:
    """documentsレコード(status=pending)を処理する。失敗時は status=error。"""
    doc_id = str(document_id)
    conn = db()
    cursor = await conn.execute(
        "SELECT collection_id, content_type FROM documents WHERE id = ?", (doc_id,)
    )
    row = await cursor.fetchone()
    if row is None:
        logger.error("ingest: document %s not found", doc_id)
        return

    await conn.execute(
        "UPDATE documents SET status = 'processing', updated_at = datetime('now') WHERE id = ?",
        (doc_id,),
    )
    await conn.commit()

    try:
        raw_path = Path(settings.data_dir) / "raw" / doc_id
        data = raw_path.read_bytes()

        chunks = _parse_chunks(row["content_type"], data)
        if not chunks:
            raise ValueError("no extractable content in document")

        is_worker_mode = settings.ingest_mode == "worker"

        vectors: list[list[float]] | None = None
        if not is_worker_mode:
            provider = get_embedding_provider()
            vectors = await provider.embed_documents([c.content for c in chunks])

        # 再取り込みに備えて既存チャンクを全消去してから書き直す
        cursor = await conn.execute("SELECT id FROM chunks WHERE document_id = ?", (doc_id,))
        old_ids = [r["id"] for r in await cursor.fetchall()]
        for old_id in old_ids:
            await conn.execute("DELETE FROM chunks_fts WHERE chunk_id = ?", (old_id,))
            await conn.execute("DELETE FROM chunk_vectors WHERE chunk_id = ?", (old_id,))
        await conn.execute("DELETE FROM chunks WHERE document_id = ?", (doc_id,))

        chunk_ids = []
        for idx, chunk in enumerate(chunks):
            chunk_id = str(uuid4())
            chunk_ids.append(chunk_id)
            await conn.execute(
                """
                INSERT INTO chunks
                    (id, document_id, collection_id, chunk_index, content, heading_path,
                     token_count)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    chunk_id,
                    doc_id,
                    row["collection_id"],
                    idx,
                    chunk.content,
                    dump_heading_path(chunk.heading_path),
                    _estimate_token_count(chunk.content),
                ),
            )
            await conn.execute(
                "INSERT INTO chunks_fts (chunk_id, content) VALUES (?, ?)",
                (chunk_id, chunk.content),
            )

        if vectors is not None:
            for chunk_id, vector in zip(chunk_ids, vectors, strict=True):
                await conn.execute(
                    "INSERT INTO chunk_vectors (chunk_id, embedding) VALUES (?, ?)",
                    (chunk_id, serialize_vector(vector)),
                )

        # workerモードではembeddingをGoワーカーに委ねるため'embedding'で止める。
        next_status = "embedding" if is_worker_mode else "ready"
        await conn.execute(
            "UPDATE documents SET status = ?, error = NULL, "
            "updated_at = datetime('now') WHERE id = ?",
            (next_status, doc_id),
        )
        await conn.commit()
    except Exception as exc:
        logger.exception("ingest: failed to process document %s", doc_id)
        await conn.rollback()
        await conn.execute(
            "UPDATE documents SET status = 'error', error = ?, "
            "updated_at = datetime('now') WHERE id = ?",
            (str(exc)[:2000], doc_id),
        )
        await conn.commit()
        raise
