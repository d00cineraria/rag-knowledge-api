"""指標計算(eval/metrics.py)のユニットテスト。外部I/Oなし。"""

import pytest
from metrics import is_relevant, mean, mrr, ndcg_at_k, recall_at_k

DOC = "employment_rules.md"


def _src(filename: str, heading_path: list[str]) -> dict:
    return {"filename": filename, "heading_path": heading_path}


def _ref(filename: str, heading_path: list[str]) -> dict:
    return {"filename": filename, "heading_path": heading_path}


def test_is_relevant_exact_match():
    ref = _ref(DOC, ["第4章 休暇", "第19条 年次有給休暇の繰越"])
    src = _src(DOC, ["第4章 休暇", "第19条 年次有給休暇の繰越"])
    assert is_relevant(src, ref)


def test_is_relevant_prefix_match_allows_deeper_retrieved_path():
    ref = _ref(DOC, ["第4章 休暇"])
    src = _src(DOC, ["第4章 休暇", "第19条 年次有給休暇の繰越"])
    assert is_relevant(src, ref)


def test_is_relevant_rejects_different_filename():
    ref = _ref(DOC, ["第4章 休暇"])
    src = _src("other.md", ["第4章 休暇"])
    assert not is_relevant(src, ref)


def test_is_relevant_rejects_when_retrieved_path_shorter_than_relevant():
    ref = _ref(DOC, ["第4章 休暇", "第19条 年次有給休暇の繰越"])
    src = _src(DOC, ["第4章 休暇"])
    assert not is_relevant(src, ref)


def test_recall_at_k_counts_distinct_relevant_hits():
    relevant = [_ref(DOC, ["第1条"]), _ref(DOC, ["第2条"])]
    sources = [_src(DOC, ["第1条"]), _src(DOC, ["第3条"])]
    assert recall_at_k(sources, relevant, k=8) == 0.5


def test_recall_at_k_respects_k_cutoff():
    relevant = [_ref(DOC, ["第1条"])]
    sources = [_src(DOC, ["第9条"]), _src(DOC, ["第8条"]), _src(DOC, ["第1条"])]
    assert recall_at_k(sources, relevant, k=2) == 0.0
    assert recall_at_k(sources, relevant, k=3) == 1.0


def test_recall_at_k_raises_for_empty_relevant():
    with pytest.raises(ValueError):
        recall_at_k([], [], k=3)


def test_mrr_reciprocal_of_first_hit_rank():
    relevant = [_ref(DOC, ["第1条"])]
    sources = [_src(DOC, ["第9条"]), _src(DOC, ["第1条"])]
    assert mrr(sources, relevant) == 0.5


def test_mrr_zero_when_no_hit():
    relevant = [_ref(DOC, ["第1条"])]
    sources = [_src(DOC, ["第9条"])]
    assert mrr(sources, relevant) == 0.0


def test_mrr_raises_for_empty_relevant():
    with pytest.raises(ValueError):
        mrr([], [])


def test_ndcg_at_k_perfect_ranking_is_one():
    relevant = [_ref(DOC, ["第1条"]), _ref(DOC, ["第2条"])]
    sources = [_src(DOC, ["第1条"]), _src(DOC, ["第2条"]), _src(DOC, ["第9条"])]
    assert ndcg_at_k(sources, relevant, k=8) == pytest.approx(1.0)


def test_ndcg_at_k_penalizes_lower_rank_hits():
    relevant = [_ref(DOC, ["第1条"])]
    sources_top = [_src(DOC, ["第1条"]), _src(DOC, ["第9条"])]
    sources_bottom = [_src(DOC, ["第9条"]), _src(DOC, ["第1条"])]
    assert ndcg_at_k(sources_top, relevant, k=8) > ndcg_at_k(sources_bottom, relevant, k=8)


def test_ndcg_at_k_zero_when_no_hit():
    relevant = [_ref(DOC, ["第1条"])]
    sources = [_src(DOC, ["第9条"])]
    assert ndcg_at_k(sources, relevant, k=8) == 0.0


def test_mean_of_empty_list_is_zero():
    assert mean([]) == 0.0


def test_mean_basic():
    assert mean([1.0, 2.0, 3.0]) == pytest.approx(2.0)
