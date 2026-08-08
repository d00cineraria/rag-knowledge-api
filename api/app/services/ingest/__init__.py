"""取り込みパイプライン（WS1が実装）。

契約: docs/contracts.md 参照。
/data/raw/{document_id} のファイルをパース→見出し考慮チャンキング→
Gemini embedding（768次元・正規化）→ chunks へ INSERT → documents.status を更新する。
"""

import logging
from pathlib import Path
from uuid import UUID

from app.config import settings
from app.db import pool

from .chunking import Chunk, chunk_markdown, chunk_pdf_pages
from .embedding import GeminiEmbedder
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
    """documentsレコード(status=pending)を読み、/data/raw/{document_id} のファイルを
    パース→チャンキング→embedding→chunksへINSERT→status=ready に更新。
    失敗時は status=error, error=メッセージ。FastAPI BackgroundTasksから呼ばれる。"""
    db = pool()
    row = await db.fetchrow(
        "SELECT collection_id, content_type FROM documents WHERE id = $1", document_id
    )
    if row is None:
        logger.error("ingest: document %s not found", document_id)
        return

    await db.execute(
        "UPDATE documents SET status = 'processing', updated_at = now() WHERE id = $1",
        document_id,
    )

    try:
        raw_path = Path(settings.data_dir) / "raw" / str(document_id)
        data = raw_path.read_bytes()

        chunks = _parse_chunks(row["content_type"], data)
        if not chunks:
            raise ValueError("no extractable content in document")

        embedder = GeminiEmbedder()
        vectors = await embedder.embed([c.content for c in chunks])

        rows = [
            (
                document_id,
                row["collection_id"],
                idx,
                chunk.content,
                chunk.heading_path,
                _estimate_token_count(chunk.content),
                vector,
            )
            for idx, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True))
        ]

        async with db.acquire() as conn, conn.transaction():
            await conn.execute("DELETE FROM chunks WHERE document_id = $1", document_id)
            await conn.executemany(
                """
                INSERT INTO chunks
                    (document_id, collection_id, chunk_index, content, heading_path,
                     token_count, embedding)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                """,
                rows,
            )

        await db.execute(
            "UPDATE documents SET status = 'ready', error = NULL, updated_at = now() WHERE id = $1",
            document_id,
        )
    except Exception as exc:
        logger.exception("ingest: failed to process document %s", document_id)
        await db.execute(
            "UPDATE documents SET status = 'error', error = $2, updated_at = now() WHERE id = $1",
            document_id,
            str(exc)[:2000],
        )
        raise
