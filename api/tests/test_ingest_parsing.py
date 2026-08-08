from io import BytesIO

from pypdf import PdfWriter

from app.services.ingest.parsing import extract_pdf_pages


def test_extract_pdf_pages_returns_one_string_per_page():
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    writer.add_blank_page(width=200, height=200)
    buf = BytesIO()
    writer.write(buf)

    pages = extract_pdf_pages(buf.getvalue())

    assert len(pages) == 2
    assert all(isinstance(p, str) for p in pages)
