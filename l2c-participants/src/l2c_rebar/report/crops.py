"""Small images of the sheets, so the engineer sees where each finding is."""

from __future__ import annotations

import io

import pymupdf
from PIL import Image, ImageDraw

from ..models import BBox

RED = (214, 39, 40)


class Cropper:
    """Renders page regions with the finding circled.  Keeps PDFs open between calls."""

    def __init__(self) -> None:
        self._docs: dict[str, pymupdf.Document] = {}
        self._thumbs: dict[tuple[str, int], Image.Image] = {}

    def close(self) -> None:
        for doc in self._docs.values():
            doc.close()
        self._docs.clear()
        self._thumbs.clear()

    def _page(self, path: str, page_no: int) -> pymupdf.Page:
        if path not in self._docs:
            self._docs[path] = pymupdf.open(path)
        return self._docs[path][page_no - 1]

    def crop(self, path: str, page_no: int, marks: list[BBox], width: float = 300.0, height: float = 190.0,
             zoom: float = 2.0) -> bytes:
        """PNG of the area around `marks` (displayed coordinates), each mark circled in red."""
        page = self._page(path, page_no)
        pw, ph = page.rect.width, page.rect.height
        x0 = min(m[0] for m in marks)
        y0 = min(m[1] for m in marks)
        x1 = max(m[2] for m in marks)
        y1 = max(m[3] for m in marks)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        width = min(max(width, (x1 - x0) + 60), pw)
        height = min(max(height, (y1 - y0) + 60), ph)
        left = min(max(cx - width / 2, 0), pw - width)
        top = min(max(cy - height / 2, 0), ph - height)
        clip = pymupdf.Rect(left, top, left + width, top + height)
        zoom = min(zoom, 1400.0 / max(width, height))
        pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), clip=clip, alpha=False)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        draw = ImageDraw.Draw(img)
        for m in marks:
            pad = 7.0
            box = [(m[0] - pad - left) * zoom, (m[1] - pad - top) * zoom, (m[2] + pad - left) * zoom,
                   (m[3] + pad - top) * zoom]
            draw.ellipse(box, outline=RED, width=max(2, int(1.6 * zoom)))
        return _png(img)

    def minimap(self, path: str, page_no: int, point: tuple[float, float], width_px: int = 300) -> bytes:
        """Thumbnail of the whole sheet with a target on the finding."""
        key = (path, page_no)
        page = self._page(path, page_no)
        scale = width_px / page.rect.width
        if key not in self._thumbs:
            pix = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False)
            self._thumbs[key] = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        img = self._thumbs[key].copy()
        draw = ImageDraw.Draw(img)
        x, y = point[0] * scale, point[1] * scale
        for r, w in ((11, 3), (3, 3)):
            draw.ellipse([x - r, y - r, x + r, y + r], outline=RED, width=w)
        draw.line([x - 18, y, x + 18, y], fill=RED, width=1)
        draw.line([x, y - 18, x, y + 18], fill=RED, width=1)
        draw.rectangle([0, 0, img.width - 1, img.height - 1], outline=(120, 120, 120))
        return _png(img)


def _png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()
