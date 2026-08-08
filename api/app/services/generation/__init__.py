"""回答生成（WS2が実装）。

契約: docs/contracts.md 参照。
settings.llm_provider に応じてGemini/Ollamaへ委譲する。プロンプト組み立て(build_prompt)は共通。
出典引用付き回答をトークン単位でyield。出典は[1][2]形式。
"""

import json
from collections.abc import AsyncIterator

import httpx
from google import genai

from app.config import settings
from app.schemas import RetrievedChunk

INSTRUCTION = (
    "与えられた出典のみに基づいて日本語で回答し、根拠箇所に[1][2]形式の引用を付ける。"
    "出典に無い内容は「出典に情報がありません」と答える。"
)

_gemini_client: genai.Client | None = None
# テスト用のhttpx.MockTransport差し替え口（Noneなら実ネットワークに接続する）
_ollama_transport: httpx.BaseTransport | None = None


def _format_source(index: int, chunk: RetrievedChunk) -> str:
    heading = " > ".join(chunk.heading_path)
    label = f"{chunk.filename} > {heading}" if heading else chunk.filename
    return f"[{index}] ({label})\n{chunk.content}"


def build_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    """出典チャンクを整形し、指示・出典・質問からなるプロンプトを組み立てる純粋関数。"""
    sources = "\n\n".join(_format_source(i, chunk) for i, chunk in enumerate(chunks, start=1))
    return f"{INSTRUCTION}\n\n# 出典\n{sources}\n\n# 質問\n{question}"


def _get_gemini_client() -> genai.Client:
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = genai.Client(api_key=settings.gemini_api_key)
    return _gemini_client


async def stream_answer(question: str, chunks: list[RetrievedChunk]) -> AsyncIterator[str]:
    prompt = build_prompt(question, chunks)
    if settings.llm_provider == "gemini":
        async for token in _stream_gemini(prompt):
            yield token
        return
    async for token in _stream_ollama(prompt):
        yield token


async def _stream_gemini(prompt: str) -> AsyncIterator[str]:
    client = _get_gemini_client()
    stream = await client.aio.models.generate_content_stream(
        model=settings.gemini_chat_model,
        contents=prompt,
    )
    async for response in stream:
        if response.text:
            yield response.text


async def _stream_ollama(prompt: str) -> AsyncIterator[str]:
    """/api/chat (stream=true) のNDJSONを1行ずつパースし、message.contentをyieldする。"""
    async with httpx.AsyncClient(
        base_url=settings.ollama_base_url, timeout=None, transport=_ollama_transport
    ) as client:
        async with client.stream(
            "POST",
            "/api/chat",
            json={
                "model": settings.ollama_chat_model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": True,
            },
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line:
                    continue
                data = json.loads(line)
                content = data.get("message", {}).get("content")
                if content:
                    yield content
                if data.get("done"):
                    break
