import hashlib

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.db import pool

_bearer = HTTPBearer(auto_error=False)


def hash_key(raw: str) -> str:
    return hashlib.sha256(raw.encode()).hexdigest()


async def seed_dev_key(dev_key: str) -> None:
    if not dev_key:
        return
    await pool().execute(
        """
        INSERT INTO api_keys (key_hash, label) VALUES ($1, 'dev')
        ON CONFLICT (key_hash) DO NOTHING
        """,
        hash_key(dev_key),
    )


async def require_api_key(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> str:
    if credentials is None:
        raise HTTPException(status_code=401, detail="Missing API key")
    row = await pool().fetchrow(
        "SELECT id FROM api_keys WHERE key_hash = $1", hash_key(credentials.credentials)
    )
    if row is None:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return str(row["id"])
