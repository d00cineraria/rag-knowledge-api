"""PDF テキスト抽出。"""

from io import BytesIO

from pypdf import PdfReader


def extract_pdf_pages(data: bytes) -> list[str]:
    """PDF バイト列からページ毎のテキストを抽出する。"""
    reader = PdfReader(BytesIO(data))
    return [page.extract_text() or "" for page in reader.pages]
