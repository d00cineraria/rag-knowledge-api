"""プロンプト生成・プロバイダ分岐(eval/llm_judge.py)のユニットテスト。

Gemini/Ollamaへの実ネットワーク呼び出しはせず、httpxはMockTransportで差し替える。
"""

import json

import httpx
import llm_judge
import pytest
from llm_judge import (
    JudgeScore,
    build_answer_relevancy_prompt,
    build_faithfulness_prompt,
    call_ollama_judge,
    judge_answer,
)


def test_build_faithfulness_prompt_includes_question_and_sources():
    prompt = build_faithfulness_prompt(
        "副業は可能ですか？",
        "はい、事前届出により可能です。",
        [
            {
                "filename": "employment_rules.md",
                "heading_path": ["第9条"],
                "content": "副業は事前届出制。",
            }
        ],
    )
    assert "副業は可能ですか？" in prompt
    assert "employment_rules.md" in prompt
    assert "副業は事前届出制。" in prompt


def test_build_faithfulness_prompt_handles_no_sources():
    prompt = build_faithfulness_prompt("資本金は？", "情報がありません。", [])
    assert "検索結果なし" in prompt


def test_build_answer_relevancy_prompt_includes_reference_answer():
    prompt = build_answer_relevancy_prompt(
        "有給の繰越上限は？", "20日です。", "20日まで繰り越し可能。"
    )
    assert "有給の繰越上限は？" in prompt
    assert "20日まで繰り越し可能。" in prompt


def test_provider_defaults_to_ollama(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    assert llm_judge._provider() == "ollama"


def test_provider_reads_env_var(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    assert llm_judge._provider() == "gemini"


async def test_call_ollama_judge_parses_json_from_format_response(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "message": {
                    "content": json.dumps({"score": 4, "reason": "おおむね妥当"})
                },
                "done": True,
            },
        )

    monkeypatch.setattr(llm_judge, "_ollama_transport", httpx.MockTransport(handler))

    result = await call_ollama_judge("採点してください", model="qwen3.5:9b")

    assert result == JudgeScore(score=4, reason="おおむね妥当")
    assert captured["body"]["model"] == "qwen3.5:9b"
    assert captured["body"]["format"] == llm_judge._JUDGE_SCHEMA
    assert captured["body"]["stream"] is False


async def test_call_ollama_judge_retries_once_then_raises_on_bad_json(monkeypatch):
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(200, json={"message": {"content": "not json"}, "done": True})

    monkeypatch.setattr(llm_judge, "_ollama_transport", httpx.MockTransport(handler))

    with pytest.raises(RuntimeError):
        await call_ollama_judge("採点してください")

    assert attempts == 2


async def test_call_ollama_judge_succeeds_on_retry(monkeypatch):
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(200, json={"message": {"content": "not json"}, "done": True})
        return httpx.Response(
            200,
            json={
                "message": {"content": json.dumps({"score": 5, "reason": "完璧"})},
                "done": True,
            },
        )

    monkeypatch.setattr(llm_judge, "_ollama_transport", httpx.MockTransport(handler))

    result = await call_ollama_judge("採点してください")

    assert result == JudgeScore(score=5, reason="完璧")
    assert attempts == 2


async def test_judge_answer_uses_ollama_by_default(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    calls = []

    async def fake_call_ollama_judge(prompt, **kwargs):
        calls.append(prompt)
        return JudgeScore(score=5, reason="ok")

    async def fail_if_called(*args, **kwargs):
        raise AssertionError("gemini judge must not be called when provider=ollama")

    monkeypatch.setattr(llm_judge, "call_ollama_judge", fake_call_ollama_judge)
    monkeypatch.setattr(llm_judge, "call_gemini_judge", fail_if_called)

    result = await judge_answer(
        "質問",
        "回答",
        [{"filename": "a.md", "heading_path": [], "content": "本文"}],
        "模範回答",
        api_key="",
    )

    assert result["faithfulness"] == JudgeScore(score=5, reason="ok")
    assert result["answer_relevancy"] == JudgeScore(score=5, reason="ok")
    assert len(calls) == 2


async def test_judge_answer_uses_gemini_when_provider_configured(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    captured = {}

    async def fake_call_gemini_judge(prompt, *, api_key, model):
        captured["api_key"] = api_key
        captured["model"] = model
        return JudgeScore(score=3, reason="gemini")

    async def fail_if_called(*args, **kwargs):
        raise AssertionError("ollama judge must not be called when provider=gemini")

    monkeypatch.setattr(llm_judge, "call_gemini_judge", fake_call_gemini_judge)
    monkeypatch.setattr(llm_judge, "call_ollama_judge", fail_if_called)

    result = await judge_answer(
        "質問",
        "回答",
        [],
        "模範回答",
        api_key="test-key",
        model="gemini-2.5-flash",
    )

    assert result["faithfulness"] == JudgeScore(score=3, reason="gemini")
    assert captured["api_key"] == "test-key"
    assert captured["model"] == "gemini-2.5-flash"
