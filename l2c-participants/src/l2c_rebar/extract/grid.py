"""Locate elements on the building grid.

On a column plan, a column has no name: it is the small filled rectangle at the
crossing of two grid lines, and a tag joined to it by a leader gives its bars.
Shop drawings name the same column by that crossing ("B-3").  This module
rebuilds the grid from the bubbles along the edges of the plan, finds the
rectangle each tag points to, and names it the way the shop drawings do.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field, replace

import pymupdf

from ..models import BBox, TextLine

_LETTER = re.compile(r"^[A-Z]{1,2}(?:\.\d)?'?$")
_NUMBER = re.compile(r"^\d{1,2}(?:\.\d)?'?$")


@dataclass
class GridAxes:
    """Grid lines of a plan view: label and position, for each direction."""

    vertical: list[tuple[str, float]] = field(default_factory=list)  # (label, x)
    horizontal: list[tuple[str, float]] = field(default_factory=list)  # (label, y)
    letters_are_vertical: bool = False  # which family (letters or numbers) labels the vertical lines

    @property
    def usable(self) -> bool:
        return len(self.vertical) >= 2 and len(self.horizontal) >= 2

    @property
    def ordered(self) -> bool:
        """True when labels run in order along each direction (A, B, C... and 1, 2, 3... or the
        reverse), as on a building grid; cells of a schedule read as "axes" seldom do."""
        def monotonic(lines: list[tuple[str, float]]) -> bool:
            values = [v for _, v in sorted((p, label_value(l)) for l, p in lines) if v is not None]
            steps = [b - a for a, b in zip(values, values[1:]) if b != a]
            return len(steps) < 2 or min(sum(d > 0 for d in steps), sum(d < 0 for d in steps)) <= 1
        return monotonic(self.vertical) and monotonic(self.horizontal)

    def coords(self, x: float, y: float) -> tuple[float, float, float, float] | None:
        """Where a point stands on the grid, whatever the scale or orientation of the sheet:
        (letter coordinate, number coordinate, letter step, number step).  Midway between
        lines 3 and 4 the number coordinate is 3.5 on the plan and on the shop drawing alike."""
        if not self.usable:
            return None
        across, down = _interpolate(self.vertical, x), _interpolate(self.horizontal, y)
        if across is None or down is None:
            return None
        (ca, sa), (cd, sd) = across, down
        return (ca, cd, sa, sd) if self.letters_are_vertical else (cd, ca, sd, sa)

    def ref(self, x: float, y: float) -> str | None:
        """Where a point stands, in the words an engineer uses to find it on the sheet: the crossing
        ("B-12") when the point is near both lines, otherwise the bay it falls in ("B-C/11-12",
        "B/11-12").  None off the grid."""
        if not self.usable or not self.ordered:
            return None
        across, down = _bay(self.vertical, x), _bay(self.horizontal, y)
        if across is None or down is None:
            return None
        letter, number = (across, down) if self.letters_are_vertical else (down, across)
        return f"{letter}-{number}" if "-" not in letter + number else f"{letter}/{number}"

    def name(self, x: float, y: float, between: float = 0.0) -> tuple[str, float] | None:
        """Grid reference of a point, letter first ("B-3"), with a confidence.

        The point takes the nearest line of each direction; it must lie within
        40% of the local line spacing, else it is not on the grid.  With `between`,
        a point further than that share of a bay from its nearest line stands on an
        intermediate line the plan does not label ("B.1" on the shop drawing): it is
        named by the lines around it, "B-C/3".
        """
        if not self.usable:
            return None
        along_x = _nearest(self.vertical, x)
        along_y = _nearest(self.horizontal, y)
        if along_x is None or along_y is None:
            return None
        (lx, ox), (ly, oy) = along_x, along_y
        if between > 0:
            lx = (_bracket(self.vertical, x) or lx) if ox > between else lx
            ly = (_bracket(self.horizontal, y) or ly) if oy > between else ly
        letter, number = (lx, ly) if self.letters_are_vertical else (ly, lx)
        sep = "/" if "-" in letter + number else "-"
        return f"{letter}{sep}{number}", round(1.0 - max(ox, oy), 2)


def label_value(label: str) -> float | None:
    """A grid label as a number, so that positions can be interpolated between lines:
    numbers as they are (3, 3.5), letters by their rank (A = 1, B = 2, B.1 = 2.1, AA = 27)."""
    m = re.fullmatch(r"(\d{1,2})(?:\.(\d))?", label)
    if m:
        return int(m.group(1)) + (int(m.group(2)) / 10 if m.group(2) else 0.0)
    m = re.fullmatch(r"([A-Z]{1,2})(?:\.(\d))?", label)
    if not m:
        return None
    rank = 0
    for char in m.group(1):
        rank = rank * 26 + (ord(char) - 64)
    return rank + (int(m.group(2)) / 10 if m.group(2) else 0.0)


def _interpolate(lines: list[tuple[str, float]], pos: float, overhang: float = 1.5) -> tuple[float, float] | None:
    """Grid coordinate of a position between labelled lines, with the local spacing of
    the labels (how much the coordinate grows over one bay).  None beyond `overhang`
    bays outside the grid."""
    valued = sorted((p, v) for label, p in lines if (v := label_value(label)) is not None)
    if len(valued) < 2:
        return None
    i = 0
    while i + 2 < len(valued) and pos > valued[i + 1][0]:
        i += 1
    (p0, v0), (p1, v1) = valued[i], valued[i + 1]
    if p1 - p0 < 1.0 or v1 == v0:
        return None
    t = (pos - p0) / (p1 - p0)
    if t < -overhang or t > 1.0 + overhang:
        return None
    return v0 + t * (v1 - v0), abs(v1 - v0)


def _bay(lines: list[tuple[str, float]], pos: float, near: float = 0.25) -> str | None:
    """Grid line a position is on ("B"), or the two lines it falls between ("B-C").  None beyond
    the outermost lines (by more than `near` of a bay)."""
    ordered = sorted(lines, key=lambda it: it[1])
    if len(ordered) < 2:
        return None
    i = 0
    while i + 2 < len(ordered) and pos > ordered[i + 1][1]:
        i += 1
    (la, a), (lb, b) = ordered[i], ordered[i + 1]
    if b - a < 1.0:
        return None
    t = (pos - a) / (b - a)
    if t < -near or t > 1.0 + near:  # beyond the outermost lines: not on the grid
        return None
    if t <= near:
        return la
    if t >= 1.0 - near:
        return lb
    return f"{la}-{lb}"


def _bracket(lines: list[tuple[str, float]], pos: float) -> str | None:
    """The two labelled lines a position falls between, lower label first ("B-C", "3-4")."""
    ordered = sorted(lines, key=lambda it: it[1])
    for (la, a), (lb, b) in zip(ordered, ordered[1:]):
        if a <= pos <= b:
            va, vb = label_value(la), label_value(lb)
            if va is None or vb is None:
                return None
            return f"{la}-{lb}" if va <= vb else f"{lb}-{la}"
    return None


def bay_key(label: str) -> tuple[str, str] | None:
    """Lower grid lines of the bay an element stands in, from either way of writing it:
    "B-C/12" (between B and C, on 12) and "B.1-12" (intermediate line B.1) both give ("B", "12")."""
    if "/" in label:
        letter, number = label.split("/", 1)
    elif "-" in label:
        letter, number = label.split("-", 1)
    else:
        return None
    lo_letter = letter.split("-")[0].split(".")[0]
    lo_number = number.split("-")[0].split(".")[0]
    return (lo_letter, lo_number) if lo_letter and lo_number else None


def _nearest(lines: list[tuple[str, float]], pos: float) -> tuple[str, float] | None:
    ordered = sorted(lines, key=lambda it: it[1])
    idx = min(range(len(ordered)), key=lambda i: abs(ordered[i][1] - pos))
    gaps = [b[1] - a[1] for a, b in zip(ordered, ordered[1:]) if b[1] - a[1] > 1.0]
    if not gaps:
        return None
    left = ordered[idx][1] - ordered[idx - 1][1] if idx > 0 else None
    right = ordered[idx + 1][1] - ordered[idx][1] if idx + 1 < len(ordered) else None
    pitch = min(g for g in (left, right) if g) if (left or right) else statistics.median(gaps)
    off = abs(ordered[idx][1] - pos) / max(pitch, 1.0)
    return (ordered[idx][0], off) if off <= 0.4 else None


def _bands(labels: list[TextLine], axis: int, tol: float) -> list[list[TextLine]]:
    """Groups of labels sharing a coordinate (axis 0: same x, a column; axis 1: same y, a row)."""
    def pos(line: TextLine) -> float:
        return (line.bbox[axis] + line.bbox[axis + 2]) / 2

    groups: list[list[TextLine]] = []
    for line in sorted(labels, key=pos):
        if groups and pos(line) - pos(groups[-1][-1]) <= tol:
            groups[-1].append(line)
        else:
            groups.append([line])
    return [g for g in groups if len({l.text.strip() for l in g}) >= 3]


def find_grid_axes(lines: list[TextLine]) -> GridAxes:
    """Read the grid from its bubbles: one family of labels (letters or numbers)
    runs along a row and names the vertical lines, the other runs down a column
    and names the horizontal lines.  Bubbles are the largest short labels."""
    # OCR may join the two pieces of a bubble with a space ("1 2" for 12).
    lines = [replace(l, text=l.text.replace(" ", "")) if l.origin == "ocr" and len(l.text) <= 6 else l for l in lines]
    letters = [l for l in lines if _LETTER.match(l.text.strip())]
    numbers = [l for l in lines if _NUMBER.match(l.text.strip())]
    if len(letters) < 3 or len(numbers) < 3:
        return GridAxes()
    big = statistics.median(sorted(l.size for l in letters + numbers)[-max(6, (len(letters) + len(numbers)) // 3):])
    letters = [l for l in letters if l.size >= 0.8 * big]
    numbers = [l for l in numbers if l.size >= 0.8 * big]

    def rows_and_columns(labels: list[TextLine]) -> tuple[int, int]:
        tol = max(4.0, 0.5 * big)
        return (sum(len(g) for g in _bands(labels, 1, tol)), sum(len(g) for g in _bands(labels, 0, tol)))

    letter_rows, letter_cols = rows_and_columns(letters)
    number_rows, number_cols = rows_and_columns(numbers)
    letters_are_vertical = letter_rows + number_cols > letter_cols + number_rows
    across, down = (letters, numbers) if letters_are_vertical else (numbers, letters)

    def collect(labels: list[TextLine], row_bands: bool) -> list[tuple[str, float]]:
        tol = max(4.0, 0.5 * big)
        positions: dict[str, list[float]] = {}
        for band in _bands(labels, 1 if row_bands else 0, tol):
            for line in band:
                coord = (line.bbox[0] + line.bbox[2]) / 2 if row_bands else (line.bbox[1] + line.bbox[3]) / 2
                positions.setdefault(line.text.strip().rstrip("'"), []).append(coord)
        # A line labelled at both ends gives two bubbles at the same position; keep labels that agree.
        return [(label, statistics.fmean(v)) for label, v in positions.items() if max(v) - min(v) <= 2 * big]

    return GridAxes(vertical=collect(across, True), horizontal=collect(down, False),
                    letters_are_vertical=letters_are_vertical)


def page_marks(page: pymupdf.Page, min_side: float = 2.5, max_side: float = 60.0) -> tuple[list[BBox], list[tuple]]:
    """Dark filled rectangles (columns drawn in plan) and short line segments (possible leaders),
    in displayed coordinates."""
    mat = page.rotation_matrix
    marks: list[BBox] = []
    segments: list[tuple[float, float, float, float]] = []
    for path in page.get_drawings():
        rect = path.get("rect")
        if rect is None:
            continue
        fill = path.get("fill")
        if fill is not None and sum(fill) < 0.6 and min_side <= min(rect.width, rect.height) and \
                max(rect.width, rect.height) <= max_side:
            r = pymupdf.Rect(rect) * mat
            r.normalize()
            marks.append((r.x0, r.y0, r.x1, r.y1))
        elif path.get("type") == "s":
            for item in path.get("items") or []:
                if item[0] == "l":
                    a, b = item[1] * mat, item[2] * mat
                    if 6.0 <= abs(a.x - b.x) + abs(a.y - b.y) <= 400.0:
                        segments.append((a.x, a.y, b.x, b.y))
    return marks, segments


def _inside(x: float, y: float, box: BBox, pad: float) -> bool:
    return box[0] - pad <= x <= box[2] + pad and box[1] - pad <= y <= box[3] + pad


def _gap(x: float, y: float, box: BBox) -> float:
    """Distance from a point to a box (0 inside)."""
    dx = max(box[0] - x, 0.0, x - box[2])
    dy = max(box[1] - y, 0.0, y - box[3])
    return (dx * dx + dy * dy) ** 0.5


def _centre(box: BBox) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def placement_modes(tags: list[BBox], marks: list[BBox], bin_size: float = 10.0) -> list[tuple[float, float]]:
    """Usual positions of a tag relative to its rectangle, most frequent first.

    A draughtsman (or the software) drops each tag at the same offset from its
    column, with a few alternatives where that spot is taken.  The offsets to
    the nearest rectangle pile up on those few values.
    """
    centres = [_centre(m) for m in marks]
    bins: dict[tuple[int, int], list[tuple[float, float]]] = {}
    for tag in tags:
        tx, ty = _centre(tag)
        if not centres:
            break
        mx, my = min(centres, key=lambda c: (c[0] - tx) ** 2 + (c[1] - ty) ** 2)
        off = (tx - mx, ty - my)
        bins.setdefault((round(off[0] / bin_size), round(off[1] / bin_size)), []).append(off)
    floor = max(3, round(0.04 * len(tags)))
    ranked = sorted((b for b in bins.values() if len(b) >= floor), key=len, reverse=True)
    return [(statistics.fmean(o[0] for o in b), statistics.fmean(o[1] for o in b)) for b in ranked[:5]]


def link_tags(tags: list[BBox], marks: list[BBox], segments: list[tuple] | None = None, reach: float = 260.0,
              rank_penalty: float = 6.0) -> list[int | None]:
    """For each tag box, the index of the rectangle it describes.

    Tags and rectangles are paired one to one, best fit first.  The fit of a
    pair is how far the tag is from one of its usual positions around the
    rectangle (`placement_modes`); where no usual position stands out, it is
    simply the distance between them.
    """
    del segments  # leaders are not needed: placement is regular enough to decide
    modes = placement_modes(tags, marks)
    pairs = []
    for t, tag in enumerate(tags):
        tx, ty = _centre(tag)
        for m, mark in enumerate(marks):
            mx, my = _centre(mark)
            if abs(tx - mx) > reach or abs(ty - my) > reach:
                continue
            off = (tx - mx, ty - my)
            distance = (off[0] ** 2 + off[1] ** 2) ** 0.5
            if modes:
                cost = min(((off[0] - mode[0]) ** 2 + (off[1] - mode[1]) ** 2) ** 0.5 + rank_penalty * k
                           for k, mode in enumerate(modes))
                cost = min(cost, distance + rank_penalty * len(modes))  # an unusual but close placement
            else:
                cost = distance
            if distance <= reach:
                pairs.append((cost, t, m))
    linked: list[int | None] = [None] * len(tags)
    taken: set[int] = set()
    for _, t, m in sorted(pairs):
        if linked[t] is None and m not in taken:
            linked[t] = m
            taken.add(m)
    return linked
