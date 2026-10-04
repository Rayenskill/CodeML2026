"""Turn the pages of a plan or a shop drawing into `Element`s."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path

import pymupdf

from ..config import Config
from ..models import SOURCE_PLAN, BBox, Bar, Element, TextLine
from ..parsing.elements import find_labels, find_levels, label_candidates, level_tokens, resolve_type
from ..parsing.rebar import find_bars, normalize_ocr
from ..pdf import ocr as ocr_mod
from ..pdf.sheets import detect_sheet
from ..pdf.text import page_text_lines
from .grid import GridAxes, find_grid_axes, link_tags, page_marks
from .layout import (Label, assign_labels, assign_row_levels, attach_spacing_lines, center, cluster_stacked,
                     flag_bar_list_rows, union)


_BARE_COUNT = re.compile(r"(\d{1,3})\s?\(\s?(\d{1,3})\s?\)")
_LIST_SEPARATOR = re.compile(r"\s*(?:,|;|&|/|\bet\b|\band\b)\s*", re.IGNORECASE)


@dataclass
class PageData:
    """Everything read from one page before interpretation."""

    path: Path
    source: str
    page_index: int
    size: tuple[float, float]
    lines: list[TextLine]
    origin: str  # text / ocr / empty
    feuillet: str
    title: str
    type_element: str
    levels: tuple[str, ...]
    sheet_conf: float = 1.0
    axes: GridAxes = field(default_factory=GridAxes)  # building grid of a plan view (column sheets)
    marks: list[BBox] = field(default_factory=list)  # dark filled rectangles: columns drawn in plan
    segments: list[tuple] = field(default_factory=list)  # short lines, among them the tag leaders
    own_lines: list[TextLine] = field(default_factory=list)  # the page's own text objects
    raw_ocr: list[TextLine] | None = None  # OCR lines as the detector boxed them, when the page was read by OCR
    ocr_only: bool = False  # OCR forced: the text objects of the page are not used


def _with_ocr(own: list[TextLine], ocr: list[TextLine], ocr_only: bool) -> list[TextLine]:
    # Keep the few real text objects of the page (the lettering-only OCR image does not show
    # them) and drop what OCR read a second time on top of them.
    return ocr if ocr_only else own + _not_covered(ocr, own)


def rejoin(page: PageData, gap: float) -> PageData:
    """The same page with the OCR pieces of each row joined (see `ocr.join_row_fragments`)."""
    if page.raw_ocr is None or gap <= 0:
        return page
    lines = _with_ocr(page.own_lines, ocr_mod.join_row_fragments(page.raw_ocr, gap), page.ocr_only)
    # Grid bubbles read in pieces ("1" and "2" for 12) are whole again: read the grid anew.
    return replace(page, lines=lines, axes=find_grid_axes(lines))


@dataclass
class DocStats:
    pages: int = 0
    pages_text: int = 0
    pages_ocr: int = 0
    pages_empty: int = 0
    by_type: Counter = field(default_factory=Counter)


def _bookmarks(doc: pymupdf.Document) -> dict[int, str]:
    out: dict[int, str] = {}
    for _lvl, title, page_no, *_ in doc.get_toc(simple=True):
        idx = page_no - 1
        if idx >= 0 and (idx not in out or len(title) > len(out[idx])):
            out[idx] = title
    return out


def _not_covered(ocr_lines: list[TextLine], text_lines: list[TextLine], pad: float = 2.0) -> list[TextLine]:
    """OCR lines whose centre does not fall on a line already known from the text layer."""
    kept = []
    for line in ocr_lines:
        cx, cy = (line.bbox[0] + line.bbox[2]) / 2, (line.bbox[1] + line.bbox[3]) / 2
        if not any(t.bbox[0] - pad <= cx <= t.bbox[2] + pad and t.bbox[1] - pad <= cy <= t.bbox[3] + pad
                   for t in text_lines):
            kept.append(line)
    return kept


def pages_needing_ocr(path: Path, cfg: Config, source: str) -> list[int]:
    if cfg.ocr == "off":
        return []
    with pymupdf.open(path) as doc:
        if cfg.ocr == "force" and source != SOURCE_PLAN:
            return list(range(doc.page_count))
        return [i for i, page in enumerate(doc)
                if sum(len(l.text.split()) for l in page_text_lines(page, cfg.ocr_min_words)) < cfg.ocr_min_words]


def read_document(
    path: Path,
    source: str,
    cfg: Config,
    cache_dir: Path | None,
    type_hint: str | None = None,
    name_levels: tuple[str, ...] = (),
) -> tuple[list[PageData], DocStats]:
    """Read every page: text layer first, OCR cache when the page has no text."""
    pages: list[PageData] = []
    stats = DocStats()
    with pymupdf.open(path) as doc:
        marks = _bookmarks(doc)
        for i, page in enumerate(doc):
            size = (page.rect.width, page.rect.height)
            lines = page_text_lines(page, cfg.ocr_min_words)
            origin = "text"
            n_words = sum(len(l.text.split()) for l in lines)
            use_ocr = cfg.ocr == "force" and source != SOURCE_PLAN
            own_lines, cached = lines, None
            if (n_words < cfg.ocr_min_words or use_ocr) and cfg.ocr != "off" and cache_dir is not None:
                cached = ocr_mod.load_cached(ocr_mod.cache_path(cache_dir, path, i, cfg))
                if cached is not None:
                    lines, origin = _with_ocr(own_lines, cached, use_ocr), "ocr"
            if not lines:
                origin = "empty"

            sheet = detect_sheet(lines, size, marks.get(i, ""))
            if source == SOURCE_PLAN:
                feuillet = sheet.feuillet if sheet else f"P{i + 1:03d}"
                title = sheet.title if sheet else ""
                type_element = resolve_type(feuillet, title)
                levels = find_levels(title)
            else:
                # A shop drawing is identified by its file; its type and level come from its name.
                feuillet = path.stem if doc.page_count == 1 else f"{path.stem} p.{i + 1}"
                title = marks.get(i, "")
                type_element = type_hint or resolve_type("", path.stem, title)
                levels = name_levels or find_levels(title)

            data = PageData(path=path, source=source, page_index=i, size=size, lines=lines, origin=origin,
                            feuillet=feuillet, title=title, type_element=type_element, levels=levels,
                            sheet_conf=1.0 if source != SOURCE_PLAN else (sheet.conf if sheet else 0.3),
                            own_lines=own_lines, raw_ocr=cached, ocr_only=use_ocr)
            # The building grid, when the page is a plan view.  OCR boxes a two-digit bubble ("12")
            # in pieces as often as whole: read the grid from the rows joined back.
            joined = cached is not None and cfg.ocr_join_gap > 0
            data.axes = find_grid_axes(
                _with_ocr(own_lines, ocr_mod.join_row_fragments(cached, cfg.ocr_join_gap), use_ocr) if joined else lines)
            if source == SOURCE_PLAN and type_element in cfg.grid_types and data.axes.usable:
                # Elements drawn as rectangles on the grid and found by their place, not by a name.
                data.marks, data.segments = page_marks(page)
            pages.append(
                data
            )
            stats.pages += 1
            stats.pages_text += origin == "text"
            stats.pages_ocr += origin == "ocr"
            stats.pages_empty += origin == "empty"
            stats.by_type[type_element] += 1
    return pages, stats


def label_vocabulary(pages: list[PageData]) -> dict[str, set[str]]:
    """Label-shaped tokens per element type, for cross-document discovery."""
    vocab: dict[str, set[str]] = defaultdict(set)
    for page in pages:
        for line in page.lines:
            vocab[page.type_element].update(label_candidates(line.text))
    return vocab


def _zone(x: float, y: float, size: tuple[float, float]) -> str:
    """Coarse sheet zone such as C4 (8 columns A-H, 6 rows 1-6), like a map grid."""
    col = min(7, max(0, int(8 * x / max(size[0], 1))))
    row = min(5, max(0, int(6 * y / max(size[1], 1))))
    return f"{'ABCDEFGH'[col]}{row + 1}"


def _tag_box(group: list[Bar], box: BBox, lines: list[TextLine]) -> BBox:
    """Extent of the whole tag a group of callouts belongs to.

    A tag stacks other lines above and below its bars (dimensions, concrete
    grade); those lines belong to the same frame, and the leader starts from it.
    """
    height = min(max(min(b.bbox[3] - b.bbox[1] for b in group), 5.0), 16.0)
    x0, y0, x1, y1 = box
    for _ in range(3):  # grow by up to three lines each way
        grown = False
        for line in lines:
            lx0, ly0, lx1, ly1 = line.bbox
            if min(x1, lx1) - max(x0, lx0) < 0.5 * min(x1 - x0, lx1 - lx0):
                continue  # not in the same column of text
            above = 0 <= y0 - ly1 <= 1.2 * height
            below = 0 <= ly0 - y1 <= 1.2 * height
            if (above or below) and (ly1 - ly0) <= 2.0 * height:
                x0, y0, x1, y1 = min(x0, lx0), min(y0, ly0), max(x1, lx1), max(y1, ly1)
                grown = True
        if not grown:
            break
    return (x0 - 4.0, y0 - 4.0, x1 + 4.0, y1 + 4.0)


def _lacks_spacing(text: str) -> bool:
    """A line holding a rebar callout that gives no spacing."""
    bars = find_bars(text)
    return bool(bars) and all(b.espacement_mm is None for b in bars)


def extract_page(page: PageData, cfg: Config, shared_labels: set[str], use_labels: bool = True,
                 header_pull: float | None = None) -> list[Element]:
    """Elements of one page.  With `use_labels` off, every callout group is its
    own unnamed element (used when plan and shop drawings share no element names)."""
    by_grid = page.axes.usable and len(page.marks) >= 3  # elements are found by their place on the grid
    bars: list[Bar] = []
    space_qty: list[bool] = []
    labels: list[Label] = []
    level_toks: list[tuple[str, float, float]] = []

    lines = [replace(l, text=normalize_ocr(l.text)) if l.origin == "ocr" else l for l in page.lines]
    if cfg.attach_spacing_lines:
        lines = attach_spacing_lines(lines, _lacks_spacing)

    group_no = 0
    for line in lines:
        is_ocr = line.origin == "ocr"
        spans: list[tuple[int, int]] = []  # characters taken by callouts (size, mark, spacing...)
        for bm in find_bars(line.text):
            spans.append((bm.start, bm.end))
            if not bm.informative and not cfg.keep_bare_diameter:
                continue
            conf = bm.conf * (0.6 + 0.4 * line.conf if is_ocr else 1.0)
            bars.append(
                Bar(diametre=bm.diametre, quantite=bm.quantite, espacement_mm=bm.espacement_mm,
                    longueur_mm=bm.longueur_mm, repere=bm.repere, role=bm.role, raw=bm.raw,
                    bbox=line.span_bbox(bm.start, bm.end), conf=round(conf, 3))
            )
            space_qty.append(bm.space_qty)
        if not spans and page.source == SOURCE_PLAN and page.type_element in cfg.count_only_types:
            # On slab plans the size is often given once in a note and each callout is a bare
            # count: "12(6)" for 12 bars, 6 of them in the column band.
            bare = _BARE_COUNT.fullmatch(line.text.strip())
            if bare and int(bare.group(1)) >= int(bare.group(2)) > 0:
                bars.append(Bar(quantite=int(bare.group(1)), raw=line.text.strip(), bbox=line.bbox,
                                conf=round(0.7 * (0.6 + 0.4 * line.conf if is_ocr else 1.0), 3)))
                space_qty.append(False)
        found = find_labels(line.text, page.type_element, extra=shared_labels,
                            grid_refs=page.type_element in cfg.grid_types) if use_labels and not by_grid else ()
        previous_end = None
        for lm in sorted(found, key=lambda m: m.start):
            if any(lm.start < end and start < lm.end for start, end in spans):
                continue  # a bar mark inside a callout (B12 in "12-15M B12 @ 300"), not an element
            # Names written as a list ("B-12, B-13 & B-14") stand for one detail drawn for all of them.
            listed = previous_end is not None and _LIST_SEPARATOR.fullmatch(line.text[previous_end: lm.start])
            if not listed:
                group_no += 1
            previous_end = lm.end
            labels.append(Label(lm.label, line.text[lm.start: lm.end], line.span_bbox(lm.start, lm.end), group_no))
        for start, end, level in level_tokens(line.text):
            cx, cy = center(line.span_bbox(start, end))
            level_toks.append((level, cx, cy))

    if not bars:
        return []

    if cfg.detect_bar_lists:
        flag_bar_list_rows(bars, space_qty)
    assign_row_levels(bars, level_toks)

    placing = [b for b in bars if not b.bordereau]
    owners = assign_labels(placing, labels, cfg.label_radius,
                           cfg.header_pulls[-1] if header_pull is None else header_pull)

    def make(element: str, group: list[Bar], anchor: tuple[float, float], labeled: bool, levels, conf,
             spot: tuple[float, float] | None = None) -> Element:
        # `anchor` is where the annotation is written, `spot` where the element itself stands.
        return Element(
            source=page.source, fichier=page.path.name, feuillet=page.feuillet, page=page.page_index + 1,
            type_element=page.type_element, element=element, x=anchor[0], y=anchor[1], bars=group,
            labeled=labeled, levels=tuple(levels), conf=round(conf, 3), page_size=page.size, path=str(page.path),
            ocr=page.origin == "ocr", grid=page.axes.coords(*(spot or anchor)),
            grid_ref=page.axes.ref(*(spot or anchor)),
        )

    elements: list[Element] = []

    # Named elements: one per (name, schedule level) on the page.  Names listed together
    # share one detail, so their callouts are pooled and each name gets all of them.
    listed: dict[int, list[Label]] = defaultdict(list)
    for lab in labels:
        listed[lab.group].append(lab)
    pooled: dict[tuple[int, str | None], list[Bar]] = defaultdict(list)
    loose: list[Bar] = []
    for bar, owner in zip(placing, owners):
        if owner is None:
            loose.append(bar)
        else:
            pooled[(owner.group, bar.level)].append(bar)
    merged: dict[tuple[str, str | None], Element] = {}
    for (group_id, level), group in pooled.items():
        names: dict[str, Label] = {}
        for lab in listed[group_id]:
            names.setdefault(lab.label, lab)
        levels = (level,) if level else page.levels
        conf = min(0.95, sum(b.conf for b in group) / len(group)) * (1.0 if page.sheet_conf >= 0.5 else 0.85)
        for label, lab in names.items():
            known = merged.get((label, level))
            if known is not None:  # the same name written twice on the page: one element
                known.bars.extend(group)
                continue
            el = make(lab.text.strip(), list(group), center(lab.bbox), True, levels, conf)
            el.label = label
            el.multiplicity = len(names)
            merged[(label, level)] = el
            elements.append(el)

    # Callouts without a name: each stacked annotation is its own element.
    groups = cluster_stacked(loose, cfg.cluster_gap)
    boxes = [union([b.bbox for b in group]) for group in groups]
    grid_names: list[tuple[str, float] | None] = [None] * len(groups)
    spots: list[tuple[float, float] | None] = [None] * len(groups)
    if by_grid and use_labels:
        # On a plan view the annotation is a tag pointing at a column: name it by the grid
        # crossing the column stands on, which is how shop drawings refer to it ("B-3").
        tags = [_tag_box(group, box, page.lines) for group, box in zip(groups, boxes)]
        for i, mark in enumerate(link_tags(tags, page.marks, page.segments)):
            if mark is not None:
                spots[i] = center(page.marks[mark])
                grid_names[i] = page.axes.name(*spots[i], between=cfg.grid_between)

    seq: Counter = Counter()
    for group, box, grid, spot in zip(groups, boxes, grid_names, spots):
        cx, cy = center(box)
        level = next((b.level for b in group if b.level), None)
        levels = (level,) if level else page.levels
        mean_conf = sum(b.conf for b in group) / len(group)
        if grid is not None:
            el = make(grid[0], group, (cx, cy), True, levels, min(0.95, mean_conf) * (0.6 + 0.4 * grid[1]), spot)
            el.label = grid[0]
            el.grid_ref = grid[0]
            elements.append(el)
            continue
        zone = _zone(cx, cy, page.size)
        seq[zone] += 1
        elements.append(make(f"ZONE-{zone}-{seq[zone]:02d}", group, (cx, cy), False, levels, 0.9 * mean_conf))

    # Bar-list rows: kept in the JSON as one record per page, never compared.
    rows = [b for b in bars if b.bordereau]
    if rows:
        box = union([b.bbox for b in rows])
        el = make("BORDEREAU", rows, center(box), False, page.levels, 0.5)
        el.bordereau = True
        elements.append(el)
    return elements


def extract_pages(pages: list[PageData], cfg: Config, shared: dict[str, set[str]],
                  label_types: set[str] | None = None, header_pulls: dict[str, float] | None = None) -> list[Element]:
    """`label_types` limits element names to the types where plan and shop
    drawings were found to share names; None trusts names everywhere.
    `header_pulls` gives, per element type, how the sheets are read (see `assign_labels`)."""
    elements: list[Element] = []
    for page in pages:
        use_labels = label_types is None or page.type_element in label_types
        elements.extend(extract_page(page, cfg, shared.get(page.type_element, set()), use_labels,
                                     (header_pulls or {}).get(page.type_element)))
    return elements


def assign_ids(elements: list[Element]) -> None:
    """Appendix A ids: `<feuillet>_<element>_<source>`, made unique."""
    seen: Counter = Counter()
    for el in elements:
        level = f"@{el.levels[0]}" if el.labeled and len(el.levels) == 1 else ""
        base = f"{el.feuillet}_{el.element}{level}_{el.source}".replace(" ", "_")
        seen[base] += 1
        el.id = base if seen[base] == 1 else f"{base}_{seen[base]}"
