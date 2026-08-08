"""run_eval.py の評価ロジックのユニットテスト。

APIレスポンス(query_fn)とLLM-as-judge(judge_fn)はどちらもフェイクの
非同期コールバックに差し替え、ネットワークI/Oなしでテストする。
"""

import json

from llm_judge import JudgeScore
from run_eval import evaluate, evaluate_question, load_golden, render_markdown

RELEVANT_QUESTION = {
    "id": "q001",
    "question": "副業は認められていますか？",
    "reference_answer": "事前届出で認められる。",
    "relevant": [
        {
            "filename": "employment_rules.md",
            "heading_path": ["第2章 服務規律", "第9条 副業・兼業"],
        }
    ],
}

OUT_OF_SCOPE_QUESTION = {
    "id": "q002",
    "question": "育児休業は何か月ですか？",
    "reference_answer": "コーパスに情報なし。",
    "relevant": [],
}


async def _fake_query_fn(question: str, top_k: int) -> dict:
    if "副業" in question:
        return {
            "answer": "事前届出で副業が認められます。",
            "sources": [
                {
                    "filename": "employment_rules.md",
                    "heading_path": ["第2章 服務規律", "第9条 副業・兼業"],
                    "content": "...",
                }
            ],
        }
    return {"answer": "情報がありません。", "sources": []}


async def _fake_judge_fn(question, answer, sources, reference_answer) -> dict:
    return {
        "faithfulness": JudgeScore(score=5, reason="ok"),
        "answer_relevancy": JudgeScore(score=4, reason="ok"),
    }


async def test_evaluate_question_computes_retrieval_metrics_when_relevant_present():
    result = await evaluate_question(
        RELEVANT_QUESTION, query_fn=_fake_query_fn, judge_fn=_fake_judge_fn, top_k=8
    )
    assert result["retrieval"]["recall@3"] == 1.0
    assert result["retrieval"]["recall@8"] == 1.0
    assert result["retrieval"]["mrr"] == 1.0
    assert result["retrieval"]["ndcg@8"] == 1.0
    assert result["faithfulness"] == 5
    assert result["answer_relevancy"] == 4


async def test_evaluate_question_skips_retrieval_metrics_for_out_of_scope():
    result = await evaluate_question(
        OUT_OF_SCOPE_QUESTION, query_fn=_fake_query_fn, judge_fn=_fake_judge_fn, top_k=8
    )
    assert result["retrieval"] is None
    assert result["faithfulness"] == 5


async def test_evaluate_aggregates_across_questions():
    report = await evaluate(
        [RELEVANT_QUESTION, OUT_OF_SCOPE_QUESTION],
        query_fn=_fake_query_fn,
        judge_fn=_fake_judge_fn,
        top_k=8,
    )
    agg = report["aggregate"]
    assert agg["num_questions"] == 2
    assert agg["num_retrieval_questions"] == 1
    assert agg["num_out_of_scope_questions"] == 1
    assert agg["recall@3"] == 1.0
    assert agg["faithfulness"] == 5.0
    assert agg["answer_relevancy"] == 4.0
    assert len(report["questions"]) == 2


def test_load_golden_parses_jsonl(tmp_path):
    path = tmp_path / "golden.jsonl"
    path.write_text(
        json.dumps(RELEVANT_QUESTION, ensure_ascii=False)
        + "\n\n"
        + json.dumps(OUT_OF_SCOPE_QUESTION, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    items = load_golden(path)
    assert [i["id"] for i in items] == ["q001", "q002"]


async def test_render_markdown_includes_metric_table_and_questions():
    report = await evaluate(
        [RELEVANT_QUESTION, OUT_OF_SCOPE_QUESTION],
        query_fn=_fake_query_fn,
        judge_fn=_fake_judge_fn,
        top_k=8,
    )
    md = render_markdown(report, top_k=8)
    assert "recall@3" in md
    assert "q001" in md
    assert "q002" in md
    assert "faithfulness" in md
