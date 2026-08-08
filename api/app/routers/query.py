import json
import time

from fastapi import APIRouter, Depends
from sse_starlette.sse import EventSourceResponse

from app.auth import require_api_key
from app.schemas import QueryRequest, QueryResponse, Source
from app.services import generation, retrieval

router = APIRouter(prefix="/v1", tags=["query"])


def _to_sources(chunks) -> list[Source]:
    return [
        Source(
            index=i + 1,
            chunk_id=c.chunk_id,
            filename=c.filename,
            heading_path=c.heading_path,
            content=c.content,
            score=c.score,
        )
        for i, c in enumerate(chunks)
    ]


@router.post("/query")
async def query(body: QueryRequest, _: str = Depends(require_api_key)):
    t0 = time.monotonic()
    chunks = await retrieval.search(body.collection_id, body.question, body.top_k)
    retrieval_ms = int((time.monotonic() - t0) * 1000)
    sources = _to_sources(chunks)

    if not body.stream:
        answer = ""
        if body.include_answer:
            async for token in generation.stream_answer(body.question, chunks):
                answer += token
        return QueryResponse(answer=answer, sources=sources)

    async def event_stream():
        yield {
            "event": "sources",
            "data": json.dumps(
                {"sources": [s.model_dump(mode="json") for s in sources]}, ensure_ascii=False
            ),
        }
        t1 = time.monotonic()
        if body.include_answer:
            async for token in generation.stream_answer(body.question, chunks):
                yield {"event": "token", "data": json.dumps({"text": token}, ensure_ascii=False)}
        yield {
            "event": "done",
            "data": json.dumps(
                {
                    "latency_ms": {
                        "retrieval": retrieval_ms,
                        "generation": int((time.monotonic() - t1) * 1000),
                    }
                }
            ),
        }

    return EventSourceResponse(event_stream())
