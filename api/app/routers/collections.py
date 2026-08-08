from fastapi import APIRouter, Depends, HTTPException

from app.auth import require_api_key
from app.db import pool
from app.schemas import Collection, CollectionCreate

router = APIRouter(prefix="/v1/collections", tags=["collections"])


@router.post("", status_code=201, response_model=Collection)
async def create_collection(
    body: CollectionCreate, _: str = Depends(require_api_key)
) -> Collection:
    row = await pool().fetchrow(
        """
        INSERT INTO collections (name, description) VALUES ($1, $2)
        ON CONFLICT (name) DO NOTHING
        RETURNING id, name, description, created_at
        """,
        body.name,
        body.description,
    )
    if row is None:
        raise HTTPException(status_code=409, detail="Collection name already exists")
    return Collection(**dict(row))


@router.get("", response_model=list[Collection])
async def list_collections(_: str = Depends(require_api_key)) -> list[Collection]:
    rows = await pool().fetch(
        "SELECT id, name, description, created_at FROM collections ORDER BY created_at"
    )
    return [Collection(**dict(r)) for r in rows]
