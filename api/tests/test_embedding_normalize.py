"""L2正規化(app.services.embedding.normalize)のユニットテスト。"""

import pytest

from app.services.embedding.normalize import l2_normalize


def test_l2_normalize_scales_to_unit_length():
    assert l2_normalize([3.0, 4.0]) == pytest.approx([0.6, 0.8])


def test_l2_normalize_zero_vector_is_unchanged():
    assert l2_normalize([0.0, 0.0]) == [0.0, 0.0]
