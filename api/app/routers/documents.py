import hashlib
from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, UploadFile

from app.auth import require_api_key
from app.config import settings
from app.db import pool
from app.schemas import DocumentStatus
from app.services import ingest

router = APIRouter(prefix="/v1", tags=["documents"])

_CONTENT_TYPES = {".md": "text/markdown", ".pdf": "application/pdf"}


@router.post("/collections/{collection_id}/documents", status_code=202)
async def upload_document(
    collection_id: UUID,
    file: UploadFile,
    background: BackgroundTasks,
    _: str = Depends(require_api_key),
) -> dict:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in _CONTENT_TYPES:
        raise HTTPException(status_code=415, detail="Only .md and .pdf are supported")

    data = await file.read()
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"Max {settings.max_upload_mb}MB")

    col = await pool().fetchrow("SELECT id FROM collections WHERE id = $1", collection_id)
    if col is None:
        raise HTTPException(status_code=404, detail="Collection not found")

    sha = hashlib.sha256(data).hexdigest()
    row = await pool().fetchrow(
        """
        INSERT INTO documents (collection_id, filename, content_type, content_sha256)
        VALUES ($1, $2, $3, $4)
        ON CONFLICT (collection_id, content_sha256) DO NOTHING
        RETURNING id
        """,
        collection_id,
        file.filename,
        _CONTENT_TYPES[suffix],
        sha,
    )
    if row is None:
        raise HTTPException(status_code=409, detail="Identical document already uploaded")

    document_id = row["id"]
    raw_dir = Path(settings.data_dir) / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / str(document_id)).write_bytes(data)

    background.add_task(ingest.process_document, document_id)
    return {"document_id": str(document_id), "status": "pending"}


@router.get("/documents/{document_id}", response_model=DocumentStatus)
async def get_document(document_id: UUID, _: str = Depends(require_api_key)) -> DocumentStatus:
    row = await pool().fetchrow(
        "SELECT id, collection_id, filename, status, error FROM documents WHERE id = $1",
        document_id,
    )
    if row is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return DocumentStatus(**dict(row))
