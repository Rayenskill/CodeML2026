"""Use the line work of a CAD plot to give OCR a clean picture of the lettering.

A sheet plotted without fonts still holds its lettering as geometry: one small
filled outline (or a few short strokes) per character.  Dimension lines, rebar,
hatching and frames are long paths.  Redrawing only the character-sized paths
on a blank page removes everything that crosses or touches the text, which is
what misleads OCR on drawings.  Nothing is recognised here; positions are
unchanged, so the result drops into the normal OCR step.
"""

from __future__ import annotations

import numpy as np
import pymupdf


def text_paths(page: pymupdf.Page, max_size: float) -> list[dict]:
    """Paths no larger than a character (`max_size` in points, on both sides)."""
    keep = []
    for path in page.get_drawings():
        rect = path.get("rect")
        if rect is None or path.get("type") == "clip":
            continue
        items = path.get("items")
        if not items:
            continue
        long_side, short_side = max(rect.width, rect.height), min(rect.width, rect.height)
        if long_side <= max_size:
            keep.append(path)
        elif short_side <= max_size and len(items) >= 12 and len(items) / long_side >= 0.3:
            # A whole string plotted as one path: long, no taller than a character, and
            # made of many segments, which a dimension line or a bar is not.
            keep.append(path)
    return keep


def _replay(shape: pymupdf.Shape, path: dict, pen: float) -> None:
    for item in path["items"]:
        kind = item[0]
        if kind == "l":
            shape.draw_line(item[1], item[2])
        elif kind == "c":
            shape.draw_bezier(item[1], item[2], item[3], item[4])
        elif kind == "re":
            shape.draw_rect(item[1])
        elif kind == "qu":
            shape.draw_quad(item[1])
    if "f" in path.get("type", ""):  # outline lettering: fill it, with a hairline so thin stems survive rasterising
        shape.finish(fill=(0, 0, 0), color=(0, 0, 0), width=0.25, even_odd=bool(path.get("even_odd", True)),
                     closePath=True)
    else:  # single-stroke lettering: redraw with a pen thick enough to be read
        shape.finish(fill=None, color=(0, 0, 0), width=max(path.get("width") or 0.0, pen),
                     closePath=bool(path.get("closePath", False)))


def lettering_image(page: pymupdf.Page, scale: float, max_size: float = 22.0, pen: float = 0.6,
                    min_paths: int = 40) -> np.ndarray | None:
    """Grayscale image of the page showing only character-sized paths, as displayed.

    Returns None when the page has too few such paths to be a vector plot (a
    scan, for instance); the caller then falls back to the plain raster.
    """
    paths = text_paths(page, max_size)
    if len(paths) < min_paths:
        return None
    canvas_doc = pymupdf.open()
    try:
        # Paths are reported in the unrotated page space: draw there, then rotate the canvas like the page.
        box = page.cropbox
        canvas = canvas_doc.new_page(width=box.width, height=box.height)
        shape = canvas.new_shape()
        for path in paths:
            _replay(shape, path, pen)
        shape.commit()
        canvas.set_rotation(page.rotation)
        pix = canvas.get_pixmap(matrix=pymupdf.Matrix(scale, scale), colorspace=pymupdf.csGRAY, alpha=False)
        return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width).copy()
    finally:
        canvas_doc.close()
