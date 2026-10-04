"""Revision comparison (bonus): what changed between two issues of a shop drawing."""

from __future__ import annotations

import tempfile
from pathlib import Path

from .config import Config
from .extract.document import assign_ids, extract_pages, pages_needing_ocr, read_document
from .models import SOURCE_SHOP, Element
from .pdf import ocr as ocr_mod


def _extract(path: Path, cfg: Config, cache_dir: Path) -> list[Element]:
    if cfg.ocr != "off" and ocr_mod.ocr_available():
        ocr_mod.ocr_pages([(path, i) for i in pages_needing_ocr(path, cfg, SOURCE_SHOP)], cfg, cache_dir)
    pages, _ = read_document(path, SOURCE_SHOP, cfg, cache_dir)
    elements = [e for e in extract_pages(pages, cfg, {}) if not e.bordereau]
    assign_ids(elements)
    return elements


def _signature(el: Element) -> list[tuple]:
    return sorted((b.diametre or "", b.quantite or 0, b.espacement_mm or 0, b.longueur_mm or 0) for b in el.bars)


def _describe(el: Element) -> dict:
    return {"element": el.element, "page": el.page, "x": round(el.x, 1), "y": round(el.y, 1),
            "armature": [b.describe() for b in el.bars]}


def diff_revisions(old: Path, new: Path, cfg: Config | None = None, cache_dir: Path | None = None) -> dict[str, list]:
    """Pair the elements of two revisions (by label, else by position on the same
    page) and sort them into added / removed / modified / unchanged."""
    cfg = cfg or Config()
    cache_dir = cache_dir or Path(tempfile.gettempdir()) / "l2c-rebar-ocr-cache"
    old_elements, new_elements = _extract(Path(old), cfg, cache_dir), _extract(Path(new), cfg, cache_dir)

    changes: dict[str, list] = {"ajoutes": [], "retires": [], "modifies": [], "inchanges": []}
    remaining = list(old_elements)
    for n in new_elements:
        match = None
        if n.label:
            match = next((o for o in remaining if o.label == n.label and o.page == n.page), None) or next(
                (o for o in remaining if o.label == n.label), None)
        if match is None:
            near = [(abs(o.x - n.x) + abs(o.y - n.y), o) for o in remaining if not o.label and o.page == n.page]
            near = [it for it in near if it[0] <= 40.0]
            match = min(near, key=lambda it: it[0])[1] if near and not n.label else None
        if match is None:
            changes["ajoutes"].append(_describe(n))
            continue
        remaining.remove(match)
        if _signature(match) == _signature(n):
            changes["inchanges"].append(_describe(n))
        else:
            changes["modifies"].append({"avant": _describe(match), "apres": _describe(n)})
    changes["retires"] = [_describe(o) for o in remaining]
    return changes
