"""LLM-as-judgeによる生成品質評価(faithfulness / answer_relevancy)。

Gemini(`gemini-2.5-flash`)に構造化出力で1〜5点のスコアを返させる。
実際のAPI呼び出しは `call_gemini_judge` 1箇所に閉じ込めてあり、
run_eval.py はテスト時にこれを差し替え可能な非同期コールバック(judge_fn)
として注入する(eval/tests/test_run_eval.py 参照)。google-genai の
importはこの関数内でのみ行うため、プロンプト生成のユニットテストは
google-genaiがインストールされていなくても実行できる。
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field


class JudgeScore(BaseModel):
    score: int = Field(ge=1, le=5)
    reason: str


def build_faithfulness_prompt(
    question: str, answer: str, sources: list[dict[str, Any]]
) -> str:
    if sources:
        context = "\n\n".join(
            f"[出典{i + 1}] {s.get('filename', '')} {s.get('heading_path', [])}\n"
            f"{s.get('content', '')}"
            for i, s in enumerate(sources)
        )
    else:
        context = "(検索結果なし)"

    return f"""あなたはRAG(検索拡張生成)システムの回答品質を評価する審査者です。
以下の「参照文書」に書かれている内容だけを根拠として、「回答」が忠実(faithful)かどうかを
1〜5の整数で評価してください。

評価基準:
5: 回答の主張がすべて参照文書の内容から直接裏付けられる
3: 一部は裏付けられるが、参照文書にない推測や一般論が混じっている
1: 参照文書に無い内容を事実であるかのように述べている(ハルシネーション)、または
   参照文書が存在しないのに何かを断定的に回答している
参照文書が「(検索結果なし)」で、回答が「情報がない」旨を正しく伝えている場合は5と評価してください。

# 質問
{question}

# 参照文書
{context}

# 回答
{answer}

score(1-5の整数)とreason(日本語で評価理由)をJSONで返してください。"""


def build_answer_relevancy_prompt(question: str, answer: str, reference_answer: str) -> str:
    return f"""あなたはRAG(検索拡張生成)システムの回答品質を評価する審査者です。
「回答」が「質問」に対してどれだけ的確に答えているかを1〜5の整数で評価してください。
参考として模範回答例も示します。

評価基準:
5: 質問の意図に過不足なく答えている
3: 部分的にしか答えていない、または冗長・的外れな内容が混じっている
1: 質問に答えていない、論点がずれている

# 質問
{question}

# 模範回答例(参考)
{reference_answer}

# 回答
{answer}

score(1-5の整数)とreason(日本語で評価理由)をJSONで返してください。"""


async def call_gemini_judge(
    prompt: str, *, api_key: str, model: str = "gemini-2.5-flash"
) -> JudgeScore:
    """Geminiに構造化出力(JSON)でスコアを問い合わせる。実ネットワークI/Oあり。"""
    import asyncio

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    response = await asyncio.to_thread(
        client.models.generate_content,
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=JudgeScore,
            temperature=0.0,
        ),
    )
    data = json.loads(response.text)
    return JudgeScore(**data)


async def judge_answer(
    question: str,
    answer: str,
    sources: list[dict[str, Any]],
    reference_answer: str,
    *,
    api_key: str,
    model: str = "gemini-2.5-flash",
) -> dict[str, JudgeScore]:
    faithfulness = await call_gemini_judge(
        build_faithfulness_prompt(question, answer, sources), api_key=api_key, model=model
    )
    answer_relevancy = await call_gemini_judge(
        build_answer_relevancy_prompt(question, answer, reference_answer),
        api_key=api_key,
        model=model,
    )
    return {"faithfulness": faithfulness, "answer_relevancy": answer_relevancy}
