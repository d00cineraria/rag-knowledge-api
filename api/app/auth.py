import hashlib
from uuid import uuid4

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.db import db

_bearer = HTTPBearer(auto_error=False)


def hash_key(raw: str) -> str:
    return hashlib.sha256(raw.encode()).hexdigest()


async def seed_dev_key(dev_key: str) -> None:
    if not dev_key:
        return
    await db().execute(
        "INSERT OR IGNORE INTO api_keys (id, key_hash, label) VALUES (?, ?, 'dev')",
        (str(uuid4()), hash_key(dev_key)),
    )
    await db().commit()


async def require_api_key(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> str:
    if credentials is None:
        raise HTTPException(status_code=401, detail="Missing API key")
    cursor = await db().execute(
        "SELECT id FROM api_keys WHERE key_hash = ?", (hash_key(credentials.credentials),)
    )
    row = await cursor.fetchone()
    if row is None:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return str(row["id"])
