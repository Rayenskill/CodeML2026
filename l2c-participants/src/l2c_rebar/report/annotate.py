"""Annotated copies of the plan and shop drawings (bonus of the rules).

Each finding is circled on the plan and on the shop drawing; the tag next to a
circle (NC-0007) is a link that opens the other document at the matching spot.
Links are relative, so the `annotes/` folder can be moved as a whole.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pymupdf

from ..models import AJOUTE, CONFORME, MANQUANT, NON_CONFORME, BBox, Element, Result

COLORS = {NON_CONFORME: (0.84, 0.15, 0.16), MANQUANT: (0.90, 0.47, 0.0), AJOUTE: (0.12, 0.40, 0.75)}


@dataclass
class Mark:
    page: int  # 1-based
    bbox: BBox  # displayed coordinates
    statut: str
    tag: str
    text: str
    target: tuple[str, int, float, float] | None = None  # (annotated file name, page, x, y)


def annotated_name(path: str) -> str:
    return f"{Path(path).stem}_annote.pdf"


def _anchor_box(el: Element) -> BBox:
    """Box to circle when no single bar is at fault: the element's callouts, else its label."""
    if el.bars:
        boxes = [b.bbox for b in el.bars]
        box = (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))
        if box[2] - box[0] <= 260 and box[3] - box[1] <= 180:
            return box
    return (el.x - 14, el.y - 8, el.x + 14, el.y + 8)


def _marks(results: list[Result]) -> dict[str, list[Mark]]:
    marks: dict[str, list[Mark]] = defaultdict(list)
    for r in results:
        if r.statut == CONFORME:
            continue
        summary = "; ".join(e.message for e in r.ecarts) or {
            MANQUANT: "Présent au plan, introuvable dans les dessins d'atelier",
            AJOUTE: "Présent à l'atelier, sans équivalent au plan",
        }.get(r.statut, "")
        text = f"{r.id} - {r.type_element} {r.element}\n{summary}\nConfiance {r.confiance:.2f}"
        plan_target = (annotated_name(r.plan.path), r.plan.page, r.plan.x, r.plan.y) if r.plan else None
        shop_target = (annotated_name(r.atelier.path), r.atelier.page, r.atelier.x, r.atelier.y) if r.atelier else None
        if r.plan:
            boxes = [e.plan_bbox for e in r.ecarts if e.plan_bbox] or [_anchor_box(r.plan)]
            for box in boxes:
                marks[r.plan.path].append(Mark(r.plan.page, box, r.statut, r.id, text, shop_target))
        if r.atelier:
            boxes = [e.atelier_bbox for e in r.ecarts if e.atelier_bbox] or [_anchor_box(r.atelier)]
            for box in boxes:
                marks[r.atelier.path].append(Mark(r.atelier.page, box, r.statut, r.id, text, plan_target))
    return marks


def annotate_project(results: list[Result], out_dir: Path) -> dict[str, Path]:
    """Write `<out_dir>/annotes/<name>_annote.pdf` for every file with findings."""
    target_dir = Path(out_dir) / "annotes"
    target_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for path, marks in _marks(results).items():
        with pymupdf.open(path) as doc:
            for mark in marks:
                _draw(doc[mark.page - 1], mark)
            dest = target_dir / annotated_name(path)
            doc.save(dest, garbage=0, deflate=True)
            written[path] = dest
    return written


def _draw(page: pymupdf.Page, mark: Mark) -> None:
    derot = page.derotation_matrix  # annotations live in the unrotated page space
    color = COLORS[mark.statut]
    x0, y0, x1, y1 = mark.bbox
    pad = 8.0
    ring = pymupdf.Rect(x0 - pad, y0 - pad, x1 + pad, y1 + pad) * derot
    ring.normalize()
    circle = page.add_circle_annot(ring)
    circle.set_colors(stroke=color)
    circle.set_border(width=2.0)
    circle.set_info(title="L2C - vérification armature", content=mark.text)
    circle.update()

    tag_box = pymupdf.Rect(x0 - pad, y0 - pad - 15, x0 - pad + 64, y0 - pad - 1)
    if tag_box.y0 < 0:
        tag_box = pymupdf.Rect(x0 - pad, y1 + pad + 1, x0 - pad + 64, y1 + pad + 15)
    tag_rect = tag_box * derot
    tag_rect.normalize()
    tag = page.add_freetext_annot(tag_rect, mark.tag, fontsize=7.5, text_color=color, fill_color=(1, 1, 1),
                                  rotate=page.rotation)
    tag.set_info(title="L2C - vérification armature", content=mark.text)
    tag.update()
    if mark.target:
        file_name, target_page, tx, ty = mark.target
        page.insert_link({"kind": pymupdf.LINK_GOTOR, "from": tag_rect, "file": file_name, "page": target_page - 1,
                          "to": pymupdf.Point(tx, ty), "zoom": 0})
