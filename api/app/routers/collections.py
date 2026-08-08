from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException

from app.auth import require_api_key
from app.db import db
from app.schemas import Collection, CollectionCreate

router = APIRouter(prefix="/v1/collections", tags=["collections"])


@router.post("", status_code=201, response_model=Collection)
async def create_collection(
    body: CollectionCreate, _: str = Depends(require_api_key)
) -> Collection:
    collection_id = str(uuid4())
    cursor = await db().execute(
        """
        INSERT OR IGNORE INTO collections (id, name, description) VALUES (?, ?, ?)
        """,
        (collection_id, body.name, body.description),
    )
    await db().commit()
    if cursor.rowcount == 0:
        raise HTTPException(status_code=409, detail="Collection name already exists")
    cursor = await db().execute(
        "SELECT id, name, description, created_at FROM collections WHERE id = ?",
        (collection_id,),
    )
    row = await cursor.fetchone()
    return Collection(**dict(row))


@router.get("", response_model=list[Collection])
async def list_collections(_: str = Depends(require_api_key)) -> list[Collection]:
    cursor = await db().execute(
        "SELECT id, name, description, created_at FROM collections ORDER BY created_at"
    )
    rows = await cursor.fetchall()
    return [Collection(**dict(r)) for r in rows]
