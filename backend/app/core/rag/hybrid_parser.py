"""Optional Docling extraction used by the production hybrid parser.

Docling is intentionally optional.  Callers should catch ``ImportError`` or
runtime conversion errors and use the pdfplumber fallback.
"""

from __future__ import annotations

from pathlib import Path


def _text(item: object) -> str:
    value = getattr(item, "text", None)
    if value:
        return " ".join(str(value).split())
    exporter = getattr(item, "export_to_markdown", None)
    if callable(exporter):
        return " ".join(str(exporter()).split())
    return ""


def _pages(item: object) -> list[int]:
    result: list[int] = []
    for provenance in getattr(item, "prov", []) or []:
        page_number = getattr(provenance, "page_no", None)
        if page_number is not None:
            result.append(int(page_number))
    return result


def extract_docling_pages(file_path: Path) -> list[str]:
    """Return Docling's ordered text grouped by source page.

    Tables are retained as blocks, but their rows are not treated as clauses;
    deterministic segmentation remains responsible for legal boundaries.
    """
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    options = PdfPipelineOptions()
    options.do_ocr = False
    options.do_table_structure = True
    converter = DocumentConverter(
        allowed_formats=[InputFormat.PDF],
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=options),
        },
    )
    document = converter.convert(str(file_path)).document

    blocks: list[tuple[int, int, str]] = []
    for order, item in enumerate(getattr(document, "texts", []) or []):
        value = _text(item)
        pages = _pages(item)
        if value and pages:
            blocks.append((min(pages), order, value))
    for order, item in enumerate(
        getattr(document, "tables", []) or [], start=len(blocks)
    ):
        value = _text(item)
        pages = _pages(item)
        if value and pages:
            blocks.append((min(pages), order, value))

    if not blocks:
        return []
    last_page = max(page for page, _, _ in blocks)
    pages = [[] for _ in range(last_page)]
    for page, order, value in sorted(blocks, key=lambda block: (block[1], block[0])):
        pages[page - 1].append(value)
    return ["\n".join(page) for page in pages]
