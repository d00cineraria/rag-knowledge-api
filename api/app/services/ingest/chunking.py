"""見出し考慮チャンキング（純粋関数）。

Markdown は見出し階層（#〜######）でセクションに分割して heading_path を付与し、
長いセクションは目安 CHUNK_SIZE 文字・CHUNK_OVERLAP 文字のオーバーラップで分割する。
PDF はページテキストを段落単位でグルーピングしたうえで同じ分割ロジックにかける。
"""

import re
from dataclasses import dataclass

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_FENCE_RE = re.compile(r"^(```|~~~)")
_BREAK_SEQUENCES = ["\n\n", "\n", "。", "．", ". ", "、", " "]


@dataclass(frozen=True)
class Section:
    heading_path: list[str]
    content: str


@dataclass(frozen=True)
class Chunk:
    heading_path: list[str]
    content: str


def parse_markdown_sections(text: str) -> list[Section]:
    """Markdown を見出し階層でセクション分割する（コードフェンス内の # は無視）。"""
    stack: list[tuple[int, str]] = []
    sections: list[Section] = []
    buffer: list[str] = []
    in_fence = False

    def flush() -> None:
        content = "\n".join(buffer).strip()
        buffer.clear()
        if content:
            sections.append(Section(heading_path=[h for _, h in stack], content=content))

    for line in text.splitlines():
        if _FENCE_RE.match(line.strip()):
            in_fence = not in_fence
            buffer.append(line)
            continue
        if not in_fence:
            m = _HEADING_RE.match(line)
            if m:
                flush()
                level = len(m.group(1))
                heading = m.group(2).strip().rstrip("#").strip()
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level, heading))
                continue
        buffer.append(line)
    flush()
    return sections


def _find_break(text: str, start: int, end: int, lookback: int) -> int:
    window_start = max(start + 1, end - lookback)
    for sep in _BREAK_SEQUENCES:
        idx = text.rfind(sep, window_start, end)
        if idx != -1:
            return idx + len(sep)
    return end


def split_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """テキストを目安 chunk_size 文字・overlap 文字オーバーラップで分割する。

    区切りは改行・句点などの自然な境界を優先し、見つからなければ強制的に区切る。
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    n = len(text)
    lookback = min(chunk_size, 200)

    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            end = _find_break(text, start, end, lookback)
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        start = max(end - overlap, start + 1)

    return chunks


def chunk_markdown(
    text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for section in parse_markdown_sections(text):
        for piece in split_text(section.content, chunk_size, overlap):
            chunks.append(Chunk(heading_path=section.heading_path, content=piece))
    return chunks


def _split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def chunk_pdf_pages(
    pages: list[str], chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> list[Chunk]:
    """ページ毎のテキストを段落単位でグルーピングし、同様の分割ロジックでチャンク化する。"""
    paragraphs: list[str] = []
    for page_text in pages:
        paragraphs.extend(_split_paragraphs(page_text))
    merged = "\n\n".join(paragraphs)
    pieces = split_text(merged, chunk_size, overlap)
    return [Chunk(heading_path=[], content=piece) for piece in pieces]
