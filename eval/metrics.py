"""検索精度指標の純粋関数実装。

契約: docs/contracts.md の「評価データ形式」節を参照。
- recall@k (k=3,8) / MRR / nDCG@8
- `relevant` と検索結果(`sources`)の突合は filename + heading_path の
  **前方一致** で判定する(sourceのheading_pathがrelevantのheading_pathを
  接頭辞として含む場合に一致とみなす)。

外部I/Oを持たないため、golden.jsonlやAPIレスポンスを一切必要とせず
そのままユニットテストできる。
"""

from __future__ import annotations

import math
from typing import Any

Source = dict[str, Any]
RelevantRef = dict[str, Any]


def is_relevant(source: Source, ref: RelevantRef) -> bool:
    """1件のsourceが1件のrelevant参照にマッチするかを判定する。

    filename一致かつ、refのheading_pathがsourceのheading_pathの
    **連続部分列**として現れれば一致（チャンカーがH1タイトルを先頭に
    含める/含めないの差異に頑健にするため、先頭固定の前方一致にしない）。
    """
    if source.get("filename") != ref.get("filename"):
        return False
    ref_path = ref.get("heading_path", [])
    src_path = source.get("heading_path", [])
    if not ref_path:
        return True
    m = len(ref_path)
    if m > len(src_path):
        return False
    return any(src_path[i : i + m] == ref_path for i in range(len(src_path) - m + 1))


def relevance_labels(sources: list[Source], relevant: list[RelevantRef]) -> list[int]:
    """各sourceについて、relevantのいずれかにマッチすれば1、しなければ0を返す。"""
    return [1 if any(is_relevant(s, r) for r in relevant) else 0 for s in sources]


def recall_at_k(sources: list[Source], relevant: list[RelevantRef], k: int) -> float:
    """上位k件の中に、relevant各項目のうち何割が(1件でも)出現したか。"""
    if not relevant:
        raise ValueError("relevant must be non-empty to compute recall")
    top = sources[:k]
    hit = sum(1 for r in relevant if any(is_relevant(s, r) for s in top))
    return hit / len(relevant)


def mrr(sources: list[Source], relevant: list[RelevantRef]) -> float:
    """最初にrelevantにヒットした順位の逆数(Mean Reciprocal Rankの1問分)。"""
    if not relevant:
        raise ValueError("relevant must be non-empty to compute MRR")
    for rank, s in enumerate(sources, start=1):
        if any(is_relevant(s, r) for r in relevant):
            return 1.0 / rank
    return 0.0


def ndcg_at_k(sources: list[Source], relevant: list[RelevantRef], k: int) -> float:
    """二値関連度によるnDCG@k。"""
    if not relevant:
        raise ValueError("relevant must be non-empty to compute nDCG")
    labels = relevance_labels(sources[:k], relevant)
    dcg = sum(label / math.log2(i + 2) for i, label in enumerate(labels))
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
