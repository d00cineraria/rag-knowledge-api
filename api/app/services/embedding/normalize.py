"""L2正規化。ingest/retrieval/embeddingプロバイダで共通に使う唯一の実装。"""


def l2_normalize(values: list[float]) -> list[float]:
    norm = sum(v * v for v in values) ** 0.5
    if norm == 0.0:
        return values
    return [v / norm for v in values]
