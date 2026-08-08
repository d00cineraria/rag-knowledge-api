"""compare.py（A/B比較）のユニットテスト。"""

from compare import build_diff_table


def _report(recall3: float, faithfulness: float | None) -> dict:
    return {
        "aggregate": {
            "recall@3": recall3,
            "recall@8": recall3,
            "mrr": recall3,
            "ndcg@8": recall3,
            "faithfulness": faithfulness,
            "answer_relevancy": faithfulness,
        }
    }


def test_build_diff_table_shows_positive_and_negative_deltas():
    table = build_diff_table(
        _report(0.5, 3.0), _report(0.7, 2.5), label_a="baseline", label_b="candidate"
    )
    assert "baseline" in table
    assert "candidate" in table
    assert "+0.200" in table
    assert "-0.500" in table


def test_build_diff_table_default_labels():
    table = build_diff_table(_report(0.5, 3.0), _report(0.5, 3.0))
    assert "| A | B |" in table
    assert "+0.000" in table


def test_build_diff_table_handles_retrieval_only_reports():
    """faithfulness/answer_relevancyがNone（--retrieval-only実行）でもクラッシュしない。"""
    table = build_diff_table(_report(0.5, None), _report(0.7, None))
    assert "+0.200" in table
    assert "| faithfulness (1-5) | - | - | - |" in table
