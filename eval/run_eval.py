"""RAG評価ランナー。

`/v1/query` (stream=false) を呼び出し、検索指標(recall@3, recall@8, MRR, nDCG@8)と
生成指標(faithfulness, answer_relevancy — Geminiによる LLM-as-judge)を算出して
`eval/results/{timestamp}/report.json` + `report.md` に出力する。

契約: docs/contracts.md「評価データ形式」節。

`evaluate()` / `evaluate_question()` は query_fn / judge_fn を注入して呼べるため、
実APIやGeminiへのネットワークI/Oなしにユニットテストできる
(eval/tests/test_run_eval.py 参照)。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

import metrics
from llm_judge import JudgeScore, judge_answer

QueryFn = Callable[[str, int], Awaitable[dict[str, Any]]]
JudgeFnType = Callable[[str, str, list[dict[str, Any]], str], Awaitable[dict[str, JudgeScore]]]

_DEFAULT_GOLDEN = Path(__file__).parent / "golden" / "golden.jsonl"
_DEFAULT_RESULTS_DIR = Path(__file__).parent / "results"


def load_golden(path: Path) -> list[dict[str, Any]]:
    items = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


async def evaluate_question(
    item: dict[str, Any], *, query_fn: QueryFn, judge_fn: JudgeFnType, top_k: int
) -> dict[str, Any]:
    response = await query_fn(item["question"], top_k)
    sources = response.get("sources", [])
    answer = response.get("answer", "")
    relevant = item.get("relevant", [])

    retrieval: dict[str, float] | None = None
    if relevant:
        retrieval = {
            "recall@3": metrics.recall_at_k(sources, relevant, 3),
            "recall@8": metrics.recall_at_k(sources, relevant, 8),
            "mrr": metrics.mrr(sources, relevant),
            "ndcg@8": metrics.ndcg_at_k(sources, relevant, 8),
        }

    judged = await judge_fn(item["question"], answer, sources, item.get("reference_answer", ""))

    return {
        "id": item["id"],
        "question": item["question"],
        "answer": answer,
        "retrieval": retrieval,
        "faithfulness": judged["faithfulness"].score,
        "faithfulness_reason": judged["faithfulness"].reason,
        "answer_relevancy": judged["answer_relevancy"].score,
        "answer_relevancy_reason": judged["answer_relevancy"].reason,
    }


async def evaluate(
    golden: list[dict[str, Any]], *, query_fn: QueryFn, judge_fn: JudgeFnType, top_k: int
) -> dict[str, Any]:
    results = [
        await evaluate_question(item, query_fn=query_fn, judge_fn=judge_fn, top_k=top_k)
        for item in golden
    ]

    with_retrieval = [r for r in results if r["retrieval"] is not None]
    aggregate = {
        "num_questions": len(results),
        "num_retrieval_questions": len(with_retrieval),
        "num_out_of_scope_questions": len(results) - len(with_retrieval),
        "recall@3": metrics.mean([r["retrieval"]["recall@3"] for r in with_retrieval]),
        "recall@8": metrics.mean([r["retrieval"]["recall@8"] for r in with_retrieval]),
        "mrr": metrics.mean([r["retrieval"]["mrr"] for r in with_retrieval]),
        "ndcg@8": metrics.mean([r["retrieval"]["ndcg@8"] for r in with_retrieval]),
        "faithfulness": metrics.mean([float(r["faithfulness"]) for r in results]),
        "answer_relevancy": metrics.mean([float(r["answer_relevancy"]) for r in results]),
    }
    return {"aggregate": aggregate, "questions": results}


def render_markdown(report: dict[str, Any], *, top_k: int) -> str:
    agg = report["aggregate"]
    lines = [
        "# RAG評価レポート",
        "",
        f"- 対象問題数: {agg['num_questions']}"
        f"（検索指標対象: {agg['num_retrieval_questions']}、"
        f"出典なし想定: {agg['num_out_of_scope_questions']}）",
        f"- top_k: {top_k}",
        "",
        "## 指標サマリ",
        "",
        "| 指標 | 値 |",
        "|---|---|",
        f"| recall@3 | {agg['recall@3']:.3f} |",
        f"| recall@8 | {agg['recall@8']:.3f} |",
        f"| MRR | {agg['mrr']:.3f} |",
        f"| nDCG@8 | {agg['ndcg@8']:.3f} |",
        f"| faithfulness (1-5) | {agg['faithfulness']:.2f} |",
        f"| answer_relevancy (1-5) | {agg['answer_relevancy']:.2f} |",
        "",
        "## 問題別内訳",
        "",
        "| id | question | recall@3 | recall@8 | MRR | nDCG@8 | faithfulness | "
        "answer_relevancy |",
        "|---|---|---|---|---|---|---|---|",
    ]

    def fmt(value: float | None) -> str:
        return f"{value:.2f}" if value is not None else "-"

    for r in report["questions"]:
        ret = r["retrieval"]
        question = r["question"].replace("|", "\\|")
        lines.append(
            f"| {r['id']} | {question} "
            f"| {fmt(ret['recall@3']) if ret else '-'} "
            f"| {fmt(ret['recall@8']) if ret else '-'} "
            f"| {fmt(ret['mrr']) if ret else '-'} "
            f"| {fmt(ret['ndcg@8']) if ret else '-'} "
            f"| {r['faithfulness']} | {r['answer_relevancy']} |"
        )
    return "\n".join(lines) + "\n"


def _real_query_fn(api_url: str, api_key: str, collection_id: str) -> QueryFn:
    async def query_fn(question: str, top_k: int) -> dict[str, Any]:
        async with httpx.AsyncClient(base_url=api_url, timeout=60.0) as client:
            resp = await client.post(
                "/v1/query",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "collection_id": collection_id,
                    "question": question,
                    "top_k": top_k,
                    "stream": False,
                },
            )
            resp.raise_for_status()
            return resp.json()

    return query_fn


def _real_judge_fn(gemini_api_key: str, model: str) -> JudgeFnType:
    async def judge_fn(
        question: str, answer: str, sources: list[dict[str, Any]], reference_answer: str
    ) -> dict[str, JudgeScore]:
        return await judge_answer(
            question, answer, sources, reference_answer, api_key=gemini_api_key, model=model
        )

    return judge_fn


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAG評価ランナー")
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--collection-id", required=True)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--golden", default=str(_DEFAULT_GOLDEN))
    parser.add_argument("--out-dir", default=str(_DEFAULT_RESULTS_DIR))
    parser.add_argument(
        "--gemini-api-key", default=None, help="未指定の場合は環境変数GEMINI_API_KEYを使用"
    )
    parser.add_argument("--judge-model", default="gemini-2.5-flash")
    return parser.parse_args(argv)


def _timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


async def main_async(args: argparse.Namespace) -> Path:
    gemini_key = args.gemini_api_key or os.environ.get("GEMINI_API_KEY", "")
    if not gemini_key:
        raise SystemExit(
            "GEMINI_API_KEY が未設定です(--gemini-api-key または環境変数で指定してください)"
        )

    golden = load_golden(Path(args.golden))
    query_fn = _real_query_fn(args.api_url, args.api_key, args.collection_id)
    judge_fn = _real_judge_fn(gemini_key, args.judge_model)

    report = await evaluate(golden, query_fn=query_fn, judge_fn=judge_fn, top_k=args.top_k)

    out_dir = Path(args.out_dir) / _timestamp()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(
        json.dumps({"top_k": args.top_k, **report}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "report.md").write_text(render_markdown(report, top_k=args.top_k), encoding="utf-8")
    print(f"report written to {out_dir}")
    return out_dir


def main() -> None:
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
