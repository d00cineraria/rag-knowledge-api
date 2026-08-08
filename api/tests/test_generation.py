"""回答生成のユニットテスト。DB・Geminiは全てモック。"""

from uuid import uuid4

from app.schemas import RetrievedChunk
from app.services import generation


def _chunk(filename="doc.md", heading_path=None, content="本文"):
    return RetrievedChunk(
        chunk_id=uuid4(),
        document_id=uuid4(),
        filename=filename,
        heading_path=heading_path or [],
        content=content,
        score=0.9,
    )


def test_build_prompt_includes_instruction_and_question():
    prompt = generation.build_prompt("有給の繰越上限は？", [_chunk()])
    assert "出典のみに基づいて日本語で回答" in prompt
    assert "[1][2]" in prompt
    assert "出典に情報がありません" in prompt
    assert "有給の繰越上限は？" in prompt


def test_build_prompt_formats_source_with_heading_path():
    chunk = _chunk(filename="rules.md", heading_path=["第2章", "2.1 休暇"], content="本文A")
    prompt = generation.build_prompt("質問", [chunk])
    assert "[1] (rules.md > 第2章 > 2.1 休暇)\n本文A" in prompt


def test_build_prompt_formats_source_without_heading_path():
    chunk = _chunk(filename="rules.md", heading_path=[], content="本文A")
    prompt = generation.build_prompt("質問", [chunk])
    assert "[1] (rules.md)\n本文A" in prompt


def test_build_prompt_numbers_multiple_sources_sequentially():
    chunks = [_chunk(filename="a.md"), _chunk(filename="b.md")]
    prompt = generation.build_prompt("質問", chunks)
    assert "[1] (a.md)" in prompt
    assert "[2] (b.md)" in prompt


class _FakeChunk:
    def __init__(self, text):
        self.text = text


class _FakeStream:
    def __init__(self, texts):
        self._texts = texts

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for text in self._texts:
            yield _FakeChunk(text)


def _install_fake_gemini(monkeypatch, texts, captured=None):
    class FakeModels:
        async def generate_content_stream(self, *, model, contents):
            if captured is not None:
                captured["model"] = model
                captured["contents"] = contents
            return _FakeStream(texts)

    class FakeAio:
        def __init__(self):
            self.models = FakeModels()

    class FakeClient:
        def __init__(self, *, api_key):
            self.aio = FakeAio()

    monkeypatch.setattr(generation, "genai", type("genai", (), {"Client": FakeClient}))
    monkeypatch.setattr(generation, "_gemini_client", None)


async def test_stream_answer_yields_tokens_from_gemini_stream(monkeypatch):
    captured = {}
    _install_fake_gemini(monkeypatch, ["回答", "の断片"], captured)

    tokens = [t async for t in generation.stream_answer("質問", [_chunk()])]

    assert tokens == ["回答", "の断片"]
    assert captured["model"] == generation.settings.gemini_chat_model
    assert "質問" in captured["contents"]


async def test_stream_answer_skips_empty_text_chunks(monkeypatch):
    _install_fake_gemini(monkeypatch, ["a", None, "", "b"])

    tokens = [t async for t in generation.stream_answer("質問", [_chunk()])]

    assert tokens == ["a", "b"]
