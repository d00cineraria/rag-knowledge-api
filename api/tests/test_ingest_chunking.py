from app.services.ingest.chunking import (
    chunk_markdown,
    chunk_pdf_pages,
    parse_markdown_sections,
    split_text,
)

_HEADING_TEXT = (
    "# 第2章 勤怠\n"
    "\n"
    "章の概要。\n"
    "\n"
    "## 2.1 有給\n"
    "\n"
    "有給休暇の説明。\n"
    "\n"
    "## 2.2 欠勤\n"
    "\n"
    "欠勤の説明。\n"
)


def test_parse_markdown_sections_builds_heading_path():
    sections = parse_markdown_sections(_HEADING_TEXT)

    assert [s.heading_path for s in sections] == [
        ["第2章 勤怠"],
        ["第2章 勤怠", "2.1 有給"],
        ["第2章 勤怠", "2.2 欠勤"],
    ]
    assert sections[0].content == "章の概要。"
    assert sections[1].content == "有給休暇の説明。"
    assert sections[2].content == "欠勤の説明。"


def test_parse_markdown_sections_content_before_any_heading_has_empty_path():
    text = "冒頭のリード文。\n\n# 第1章\n\n本文。\n"

    sections = parse_markdown_sections(text)

    assert sections[0].heading_path == []
    assert sections[0].content == "冒頭のリード文。"
    assert sections[1].heading_path == ["第1章"]


def test_parse_markdown_sections_ignores_headings_inside_code_fence():
    text = "# 見出し\n\n```\n# これは見出しではない\n```\n\n本文。\n"

    sections = parse_markdown_sections(text)

    assert len(sections) == 1
    assert sections[0].heading_path == ["見出し"]
    assert "# これは見出しではない" in sections[0].content


def test_split_text_returns_whole_text_when_short():
    assert split_text("短い文章。", chunk_size=800, overlap=150) == ["短い文章。"]


def test_split_text_returns_empty_list_for_blank_text():
    assert split_text("   \n\n  ") == []


def test_split_text_splits_and_overlaps_without_natural_boundary():
    text = "a" * 250
    chunks = split_text(text, chunk_size=100, overlap=20)

    assert len(chunks) == 3
    assert chunks[0] == text[0:100]
    assert chunks[1] == text[80:180]
    assert chunks[2] == text[160:250]
    # 隣接チャンクは overlap 分だけ重複する
    assert chunks[0][-20:] == chunks[1][:20]
    assert chunks[1][-20:] == chunks[2][:20]


def test_split_text_prefers_sentence_boundary():
    sentence = "これはテスト文です。"
    text = sentence * 30

    chunks = split_text(text, chunk_size=100, overlap=10)

    assert len(chunks) > 1
    for piece in chunks[:-1]:
        assert piece.endswith("。")


def test_chunk_markdown_splits_long_section_and_keeps_heading_path():
    long_body = "これは長いセクションの本文です。" * 60
    text = f"# 見出し\n\n{long_body}\n"

    chunks = chunk_markdown(text, chunk_size=200, overlap=40)

    assert len(chunks) > 1
    assert all(c.heading_path == ["見出し"] for c in chunks)


def test_chunk_pdf_pages_groups_paragraphs_across_pages():
    pages = ["段落1\n\n段落2", "段落3"]

    chunks = chunk_pdf_pages(pages, chunk_size=1000, overlap=0)

    assert len(chunks) == 1
    assert chunks[0].heading_path == []
    assert "段落1" in chunks[0].content
    assert "段落2" in chunks[0].content
    assert "段落3" in chunks[0].content


def test_chunk_pdf_pages_splits_long_content():
    pages = ["これはPDFの段落です。" * 60]

    chunks = chunk_pdf_pages(pages, chunk_size=200, overlap=40)

    assert len(chunks) > 1
    assert all(c.heading_path == [] for c in chunks)
