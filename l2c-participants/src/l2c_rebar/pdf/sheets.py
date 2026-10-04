"""Find the sheet number (feuillet) and title of a drawing page."""

from __future__ import annotations

from dataclasses import dataclass

from ..models import TextLine
from ..parsing.elements import find_sheet_ids


@dataclass
class SheetInfo:
    feuillet: str
    title: str
    conf: float


def detect_sheet(lines: list[TextLine], page_size: tuple[float, float], bookmark: str = "") -> SheetInfo | None:
    """Sheet number from the PDF bookmark when it has one, else from the title block.

    A page usually cites several sheets ("VOIR S-501"); the one in the title
    block is the largest and sits closest to the bottom-right corner.
    """
    ids = find_sheet_ids(bookmark or "")
    if ids:
        return SheetInfo(ids[0][2], bookmark.strip(), 0.95)

    w, h = page_size
    best: tuple[float, str, TextLine] | None = None
    max_size = max((l.size for l in lines), default=1.0)
    for line in lines:
        for _, _, sheet in find_sheet_ids(line.text):
            cx = (line.bbox[0] + line.bbox[2]) / 2
            cy = (line.bbox[1] + line.bbox[3]) / 2
            corner = 1.0 - (((w - cx) / max(w, 1)) ** 2 + ((h - cy) / max(h, 1)) ** 2) ** 0.5 / 1.414
            alone = 1.0 if len(line.text.strip()) <= len(sheet) + 2 else 0.0
            score = 0.5 * (line.size / max_size) + 0.35 * corner + 0.15 * alone
            if best is None or score > best[0]:
                best = (score, sheet, line)
    if best is None:
        return None
    score, sheet, line = best
    return SheetInfo(sheet, _title_near(lines, line), round(min(0.9, 0.4 + score / 2), 2))


def _title_near(lines: list[TextLine], anchor: TextLine, reach: float = 320.0) -> str:
    """Text of the title block around the sheet number (used to read the level)."""
    ax = (anchor.bbox[0] + anchor.bbox[2]) / 2
    ay = (anchor.bbox[1] + anchor.bbox[3]) / 2
    near = []
    for line in lines:
        cx = (line.bbox[0] + line.bbox[2]) / 2
        cy = (line.bbox[1] + line.bbox[3]) / 2
        if abs(cx - ax) < reach and abs(cy - ay) < reach and line is not anchor and line.size >= 0.5 * anchor.size:
            near.append((-line.size, cy, line.text))
    near.sort()
    return " | ".join(t for _, _, t in near[:4])
