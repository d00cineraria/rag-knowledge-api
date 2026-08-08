"""取り込みパイプライン（WS1が実装）。

契約: docs/contracts.md 参照。
/data/raw/{document_id} のファイルをパース→見出し考慮チャンキング→
Gemini embedding（768次元・正規化）→ chunks へ INSERT → documents.status を更新する。
"""

from uuid import UUID


async def process_document(document_id: UUID) -> None:
    raise NotImplementedError("WS1: ingest pipeline not implemented yet")
