"""Tunable settings and domain vocabulary, in one place.

Everything a new project might need to adjust lives here; a JSON file with the
same field names can override the defaults (`--config`).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, fields
from pathlib import Path

FONDATION = "fondation"
POUTRE = "poutre"
MUR = "mur de refend"
COLONNE = "colonne"
DALLE = "dalle"
AUTRE = "autre"
ELEMENT_TYPES = (FONDATION, POUTRE, MUR, COLONNE, DALLE)

# Sheet series given by the rules: S-100 foundation, S-300 beam, S-400 shear
# wall, S-500 column, S-600 slab.
SERIES_TYPES = {1: FONDATION, 3: POUTRE, 4: MUR, 5: COLONNE, 6: DALLE}

# Keywords (accent-stripped, upper case) that reveal the element type in a
# folder name, a file name or a sheet title.  Order matters: first hit wins.
TYPE_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    (MUR, ("REFEND", "CISAILLEMENT", "CISAIL", "CONCENTRATION", "NOYAU", "MUR")),
    (COLONNE, ("COLONNE", "POTEAU")),
    (POUTRE, ("POUTRE", "LINTEAU")),
    (FONDATION, ("FONDATION", "SEMELLE", "RADIER", "PIEU", "EMPATTEMENT", "FND")),
    (DALLE, ("DALLE", "PLANCHER", "ESCALIER")),
]

# Element-label prefixes commonly used per type (C-12, S3, P-4, MR2, ...).
TYPE_PREFIXES: dict[str, frozenset[str]] = {
    COLONNE: frozenset({"C", "COL", "CO", "PC", "CB", "K"}),
    FONDATION: frozenset({"S", "SE", "SF", "SI", "SC", "SEM", "R", "RAD", "RA", "E", "F", "MF", "P", "PI", "SR", "BP"}),
    POUTRE: frozenset({"P", "PO", "PT", "B", "BM", "L", "LT", "PL", "PP", "PB", "PR"}),
    MUR: frozenset({"MR", "M", "MC", "R", "W", "SW", "ZC", "NR", "CC", "C", "N", "MUR"}),
    DALLE: frozenset({"D", "DA", "BD", "DP", "BF"}),
}

# Prefixes that look like labels but never are.
LABEL_STOP = frozenset(
    {"NIV", "SS", "RDC", "TYP", "TYPE", "DET", "REV", "MPA", "KPA", "KN", "MM", "HSS", "W", "L", "T", "B", "H", "V",
     "A", "X", "Y", "DE", "LE", "LA", "DU", "AU", "ET", "NO", "FY", "FC", "EL", "ELEV", "PAGE", "AXE", "VOIR", "ISO"}
)


@dataclass
class Config:
    # --- OCR (pages without a vector text layer) ---
    ocr: str = "auto"  # auto: only pages without text; off: never; force: every shop page
    # Recognition model.  v5 with the English recogniser read 76% of callouts exactly on a
    # benchmark page, against 57% for v4.  v4 ships inside the rapidocr-onnxruntime wheel and
    # is the automatic fallback when the newer models cannot be loaded.
    ocr_model: str = "v5"  # v4 / v5 / v6 (PP-OCR generation of the recogniser)
    ocr_det_model: str = ""  # generation of the text detector; empty = same as ocr_model
    ocr_det_type: str = "mobile"  # mobile / server (v5), tiny / small / medium (v6)
    ocr_rec_type: str = "mobile"
    ocr_rec_lang: str = "en"  # en, ch (Chinese + Latin), latin
    ocr_model_dir: str = ""  # folder holding the model files; empty = rapidocr's own cache
    ocr_dpi: int = 300  # 300 read 85% of benchmark callouts exactly, 200 read 79% and is about 25% faster
    ocr_lettering_only: bool = True  # on vector plots, show OCR only the character-sized paths (no crossing lines)
    ocr_glyph_max_pt: float = 22.0  # largest path still treated as a character, in points
    ocr_tile: int = 1600
    ocr_overlap: int = 400
    ocr_min_words: int = 25  # fewer words than this -> the page has no usable text layer
    ocr_min_score: float = 0.5
    ocr_join_gap: float = 1.5  # OCR pieces of one row closer than this many text heights are one line (0: off)
    ocr_thicken: int = 0  # pixels added around dark strokes before OCR (helps hairline CAD lettering)
    ocr_gpu: str = "auto"  # auto: use the local GPU when onnxruntime-directml is installed; off: CPU only
    workers: int = field(default_factory=lambda: max(1, min(6, (os.cpu_count() or 2) // 2)))

    # --- Extraction ---
    label_radius: float = 220.0  # max distance (pt) from a callout to its element label
    cluster_gap: float = 1.4  # stacked callouts closer than gap x text height form one annotation
    keep_bare_diameter: bool = False  # keep callouts that only state a diameter
    min_shared_labels: int = 2  # element names are used for a type only if this many appear on both sides
    detect_bar_lists: bool = False  # set aside stacked "24 15M ..." rows as a bar list (bordereau)
    attach_spacing_lines: bool = True  # a spacing written on the line after a callout ("@12"") belongs to it
    compare_other_types: bool = False  # also compare sheets outside the five series
    count_only_types: tuple[str, ...] = (DALLE,)  # plan sheets where a bare "12(6)" is a bar count
    grid_types: tuple[str, ...] = (COLONNE,)  # element types located by grid crossing ("B-3") on plan views
    # A column this many bays off its nearest line stands on an unlabelled line ("B-C/3", matched to a shop
    # "B.1-3").  Off (0): on the development projects such columns sit within 0.25 bay of a labelled line,
    # like columns merely drawn off-centre, so position cannot tell them apart.
    grid_between: float = 0.0

    # --- Comparison ---
    spacing_tol_mm: int = 3
    spacing_tol_ratio: float = 0.02  # metric vs imperial rounding: 300 mm and 12 in (305 mm) are the same spacing
    length_tol_mm: int = 25
    position_tol: float = 0.18  # normalised distance for position-aware pairing
    position_bays: float = 0.5  # callouts of a plan view are paired within this many grid bays
    recover_from_neighbours: bool = False  # let a neighbouring named element excuse a missing bar
    min_sheet_coverage: float = 0.5  # below this share of a sheet's elements found, report the sheet, not each element
    min_label_pairs: int = 5  # with at least this many pairs made by name for a type...
    header_pulls: tuple[float, ...] = (0.1, 0.35)  # readings tried for named elements: schedule, details
    min_label_agreement: float = 0.5  # ...fewer of their bars confirmed than this share means the pairing is wrong
    ocr_confidence_factor: float = 0.8  # confidence kept by a result resting on OCR text (on top of the OCR score)
    trusted_agreement: float = 0.9  # below this share of paired bars confirmed, findings by name lose confidence

    # --- Report ---
    crops: bool = True  # image extracts of each non-conformity
    annotate: bool = True  # annotated copies of the PDFs
    max_cards: int = 150  # non-conformities shown with image extracts
    max_detail_rows: int = 120  # rows per "missing" / "added" table and per sheet
    max_index_rows: int = 600  # rows of the list of all findings at the start of the report

    @classmethod
    def load(cls, path: str | os.PathLike | None) -> "Config":
        cfg = cls()
        if path:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            known = {f.name for f in fields(cls)}
            unknown = set(data) - known
            if unknown:
                raise ValueError(f"Unknown config keys: {sorted(unknown)}")
            for k, v in data.items():
                setattr(cfg, k, v)
        return cfg
