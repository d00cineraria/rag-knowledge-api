"""評価用コーパスをAPIへ投入するセットアップスクリプト。

コレクション作成(既存なら再利用) → `eval/corpus/*.md` を一括アップロード →
各文書が ready(またはerror) になるまでポーリングし、コレクションIDを標準出力に出す。
出力されたIDはそのまま `run_eval.py --collection-id` に渡せる。
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

import httpx

_CORPUS_DIR = Path(__file__).parent / "corpus"
_POLL_INTERVAL_SEC = 2.0


async def _get_or_create_collection(client: httpx.AsyncClient, name: str, description: str) -> str:
    resp = await client.post("/v1/collections", json={"name": name, "description": description})
    if resp.status_code == 201:
        return resp.json()["id"]
    if resp.status_code == 409:
        list_resp = await client.get("/v1/collections")
        list_resp.raise_for_status()
        for c in list_resp.json():
            if c["name"] == name:
                return c["id"]
        raise RuntimeError(f"collection '{name}' reported as existing but not found in list")
    resp.raise_for_status()
    raise RuntimeError("unreachable")  # pragma: no cover


async def _upload_document(client: httpx.AsyncClient, collection_id: str, path: Path) -> str:
    content_type = "text/markdown" if path.suffix == ".md" else "application/pdf"
    with path.open("rb") as f:
        resp = await client.post(
            f"/v1/collections/{collection_id}/documents",
            files={"file": (path.name, f, content_type)},
        )
    resp.raise_for_status()
    return resp.json()["document_id"]


async def _wait_ready(client: httpx.AsyncClient, document_id: str, *, timeout_sec: float) -> dict:
    deadline = time.monotonic() + timeout_sec
    while True:
        resp = await client.get(f"/v1/documents/{document_id}")
        resp.raise_for_status()
        status = resp.json()
        if status["status"] in ("ready", "error"):
            return status
        if time.monotonic() > deadline:
            raise TimeoutError(f"document {document_id} did not become ready within {timeout_sec}s")
        await asyncio.sleep(_POLL_INTERVAL_SEC)


async def setup_corpus(
    *, api_url: str, api_key: str, collection_name: str, corpus_dir: Path, timeout_sec: float
) -> str:
    async with httpx.AsyncClient(
        base_url=api_url, headers={"Authorization": f"Bearer {api_key}"}, timeout=60.0
    ) as client:
        collection_id = await _get_or_create_collection(
            client, collection_name, "WS3評価基盤コーパス(eval/setup_corpus.pyで投入)"
        )

        doc_files = sorted(p for p in corpus_dir.glob("*.md") if p.name != "SOURCES.md")
        if not doc_files:
            raise RuntimeError(f"no corpus documents found in {corpus_dir}")

        document_ids = [await _upload_document(client, collection_id, p) for p in doc_files]

        for document_id, path in zip(document_ids, doc_files, strict=True):
            status = await _wait_ready(client, document_id, timeout_sec=timeout_sec)
            if status["status"] == "error":
                raise RuntimeError(f"{path.name} failed to ingest: {status.get('error')}")
            print(f"ready: {path.name} ({document_id})", file=sys.stderr)

        return collection_id


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="評価用コーパスをAPIへ投入する")
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--collection-name", default="ws3-eval-corpus")
    parser.add_argument("--corpus-dir", default=str(_CORPUS_DIR))
    parser.add_argument("--timeout-sec", type=float, default=180.0)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    collection_id = asyncio.run(
        setup_corpus(
            api_url=args.api_url,
            api_key=args.api_key,
            collection_name=args.collection_name,
            corpus_dir=Path(args.corpus_dir),
            timeout_sec=args.timeout_sec,
        )
    )
    print(collection_id)


if __name__ == "__main__":
    main()
