"""Text lines of a PDF page, in displayed coordinates (top-left origin, points).

PyMuPDF reports text in the *unrotated* page space; every box is pushed through
`page.rotation_matrix` so that x, y match what an engineer sees on screen and
what Appendix A of the rules asks for.
"""

from __future__ import annotations

import pymupdf

from ..models import TextLine


def text_layer_lines(page: pymupdf.Page) -> list[TextLine]:
    mat = page.rotation_matrix
    lines: list[TextLine] = []
    for block in page.get_text("dict").get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            text = _join_spans(line["spans"], line["dir"])
            stripped = text.strip()
            if not stripped:
                continue
            rect = pymupdf.Rect(line["bbox"]) * mat
            rect.normalize()
            dx, dy = line["dir"]
            # Rotate the reading direction with the page (no translation).
            rdx = dx * mat.a + dy * mat.c
            rdy = dx * mat.b + dy * mat.d
            size = max((s.get("size", 8.0) for s in line["spans"]), default=8.0)
            lines.append(
                TextLine(text=stripped, bbox=(rect.x0, rect.y0, rect.x1, rect.y1), size=size, dx=rdx, dy=rdy)
            )
    return lines


def _join_spans(spans: list[dict], direction: tuple[float, float]) -> str:
    """Concatenate spans, restoring a space where two spans are visibly apart."""
    out = ""
    prev = None
    horizontal = abs(direction[0]) >= abs(direction[1])
    for span in spans:
        text = span.get("text", "")
        if prev is not None and out and not out[-1].isspace() and text and not text[0].isspace():
            a, b = prev["bbox"], span["bbox"]
            gap = (b[0] - a[2]) if horizontal else min(abs(b[1] - a[3]), abs(a[1] - b[3]))
            if gap > 0.25 * max(span.get("size", 8.0), 1.0):
                out += " "
        out += text
        prev = span
    return out


def shx_comment_lines(page: pymupdf.Page) -> list[TextLine]:
    """Text that AutoCAD keeps beside stroke lettering.

    PDF has no SHX fonts, so AutoCAD plots that lettering as line work; unless
    told otherwise it also stores each string as a comment labelled "AutoCAD SHX
    Text", with its rectangle.  When a plot kept them, the page needs no OCR.
    """
    mat = page.rotation_matrix  # annotation rectangles are in the unrotated page space
    lines: list[TextLine] = []
    for annot in page.annots() or []:
        info = annot.info or {}
        if not any("SHX" in str(info.get(key, "")) for key in ("title", "subject", "name")):
            continue
        text = str(info.get("content", "")).strip()
        if not text:
            continue
        rect = pymupdf.Rect(annot.rect) * mat
        rect.normalize()
        vertical = rect.height > 1.5 * rect.width and len(text) > 1
        lines.append(
            TextLine(text=text, bbox=(rect.x0, rect.y0, rect.x1, rect.y1),
                     size=rect.width if vertical else rect.height, dx=0.0 if vertical else 1.0,
                     dy=-1.0 if vertical else 0.0)
        )
    return lines


def page_text_lines(page: pymupdf.Page, min_words: int) -> list[TextLine]:
    """Exact text of a page: its text layer, completed by AutoCAD's SHX comments when the layer is thin."""
    lines = text_layer_lines(page)
    if sum(len(l.text.split()) for l in lines) < min_words:
        lines += shx_comment_lines(page)
    return lines
