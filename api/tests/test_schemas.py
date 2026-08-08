"""DB不要のユニットテスト（CIの基本ゲート）。"""

from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.auth import hash_key
from app.schemas import QueryRequest, RetrievedChunk


def test_hash_key_is_deterministic_sha256_hex():
    assert hash_key("abc") == hash_key("abc")
    assert len(hash_key("abc")) == 64


def test_query_request_defaults():
    req = QueryRequest(collection_id=uuid4(), question="有給の繰越上限は？")
    assert req.top_k == 8
    assert req.stream is True


def test_query_request_rejects_empty_question():
    with pytest.raises(ValidationError):
        QueryRequest(collection_id=uuid4(), question="")


def test_query_request_rejects_out_of_range_top_k():
    with pytest.raises(ValidationError):
        QueryRequest(collection_id=uuid4(), question="q", top_k=100)


def test_retrieved_chunk_roundtrip():
    c = RetrievedChunk(
        chunk_id=uuid4(),
        document_id=uuid4(),
        filename="doc.md",
        heading_path=["第2章", "2.1"],
        content="本文",
        score=0.5,
    )
    assert c.model_dump(mode="json")["heading_path"] == ["第2章", "2.1"]
