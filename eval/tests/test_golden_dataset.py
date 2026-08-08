"""golden.jsonl / corpus 自体の整合性を検証する回帰テスト。"""

import json
from pathlib import Path

_GOLDEN_PATH = Path(__file__).parent.parent / "golden" / "golden.jsonl"
_CORPUS_DIR = Path(__file__).parent.parent / "corpus"


def _load() -> list[dict]:
    with _GOLDEN_PATH.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_golden_dataset_has_reasonable_number_of_questions():
    items = _load()
    assert 20 <= len(items) <= 50


def test_golden_dataset_ids_are_unique():
    items = _load()
    ids = [i["id"] for i in items]
    assert len(ids) == len(set(ids))


def test_golden_dataset_items_have_required_fields():
    items = _load()
    for item in items:
        assert item["id"]
        assert item["question"]
        assert item["reference_answer"]
        assert "relevant" in item
        for ref in item["relevant"]:
            assert "filename" in ref
            assert "heading_path" in ref
            assert isinstance(ref["heading_path"], list)


def test_golden_dataset_includes_multi_chunk_and_out_of_scope_questions():
    items = _load()
    assert any(len(i["relevant"]) >= 2 for i in items), "複数チャンク参照問題が見つからない"
    assert any(len(i["relevant"]) == 0 for i in items), "「出典に無い」問題が見つからない"


def test_golden_dataset_references_only_corpus_filenames():
    corpus_files = {p.name for p in _CORPUS_DIR.glob("*.md") if p.name != "SOURCES.md"}
    items = _load()
    for item in items:
        for ref in item["relevant"]:
            assert ref["filename"] in corpus_files
