import hashlib
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, UploadFile

from app.auth import require_api_key
from app.config import settings
from app.db import db
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

    cursor = await db().execute(
        "SELECT id FROM collections WHERE id = ?", (str(collection_id),)
    )
    if await cursor.fetchone() is None:
        raise HTTPException(status_code=404, detail="Collection not found")

    sha = hashlib.sha256(data).hexdigest()
    document_id = str(uuid4())
    cursor = await db().execute(
        """
        INSERT OR IGNORE INTO documents (id, collection_id, filename, content_type, content_sha256)
        VALUES (?, ?, ?, ?, ?)
        """,
        (document_id, str(collection_id), file.filename, _CONTENT_TYPES[suffix], sha),
    )
    await db().commit()
    if cursor.rowcount == 0:
        raise HTTPException(status_code=409, detail="Identical document already uploaded")

    raw_dir = Path(settings.data_dir) / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / document_id).write_bytes(data)

    background.add_task(ingest.process_document, UUID(document_id))
    return {"document_id": document_id, "status": "pending"}


@router.get("/documents/{document_id}", response_model=DocumentStatus)
async def get_document(document_id: UUID, _: str = Depends(require_api_key)) -> DocumentStatus:
    cursor = await db().execute(
        "SELECT id, collection_id, filename, status, error FROM documents WHERE id = ?",
        (str(document_id),),
    )
    row = await cursor.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return DocumentStatus(**dict(row))
