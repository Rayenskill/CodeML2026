"""A made-up project with known non-conformities.

Nothing here comes from a real drawing.  The fake project has the folder layout
of the challenge data, one sheet per element type, and five planted
discrepancies, so the pipeline can be tested and demonstrated without touching
confidential documents.
"""

from __future__ import annotations

from pathlib import Path

import pymupdf

PAGE = (1224.0, 792.0)

# What `build_project` plants; tests assert that each one is found.
PLANTED = [
    {"element": "S3", "type": "fondation", "attribut": "quantite", "plan": 8, "atelier": 6},
    {"element": "P2", "type": "poutre", "attribut": "espacement_mm", "plan": 200, "atelier": 250},
    {"element": "MR2", "type": "mur de refend", "attribut": "diametre", "plan": "25M", "atelier": "20M"},
    {"element": "C3", "type": "colonne", "attribut": "quantite", "plan": 8, "atelier": 6},
    {"element": "dalle", "type": "dalle", "attribut": "quantite", "plan": 10, "atelier": 8},
]
MISSING = "C4"  # in the plan (RDC), absent from the shop drawings
ADDED = "C9"  # in the shop drawings, absent from the plan


def _text(page: pymupdf.Page, x: float, y: float, text: str, size: float = 9.0) -> None:
    page.insert_text((x, y), text, fontsize=size)


def _sheet(doc: pymupdf.Document, number: str, title: str) -> pymupdf.Page:
    page = doc.new_page(width=PAGE[0], height=PAGE[1])
    page.draw_rect(pymupdf.Rect(20, 20, PAGE[0] - 20, PAGE[1] - 20), width=1)
    page.draw_rect(pymupdf.Rect(PAGE[0] - 300, PAGE[1] - 120, PAGE[0] - 20, PAGE[1] - 20), width=1)
    _text(page, PAGE[0] - 290, PAGE[1] - 90, title, 11)
    _text(page, PAGE[0] - 290, PAGE[1] - 40, number, 22)
    return page


def _plan(path: Path) -> None:
    doc = pymupdf.open()

    page = _sheet(doc, "S-101", "FONDATIONS - TABLEAU DES SEMELLES")
    _text(page, 100, 110, "SEMELLE        DIMENSIONS        ARMATURE")
    for i, bars in enumerate(["8-20M C.S. INF.", "10-20M C.S. INF.", "8-20M C.S. INF.", "12-25M C.S. INF."]):
        y = 150 + 40 * i
        _text(page, 100, y, f"S{i + 1}")
        _text(page, 200, y, "2000x2000x500")
        _text(page, 380, y, bars)

    page = _sheet(doc, "S-301", "POUTRES - TABLEAU DES POUTRES")
    for i, (top, bottom, stirrups) in enumerate([("3-25M SUP.", "4-25M INF.", "ETRIERS 10M @ 200"),
                                                 ("2-20M SUP.", "3-25M INF.", "ETRIERS 10M @ 200"),
                                                 ("3-20M SUP.", "3-20M INF.", "ETRIERS 10M @ 150")]):
        y = 150 + 50 * i
        _text(page, 100, y, f"P{i + 1}")
        _text(page, 200, y, "400x600")
        _text(page, 320, y, top)
        _text(page, 440, y, bottom)
        _text(page, 560, y, stirrups)

    page = _sheet(doc, "S-401", "MURS DE REFEND - ELEVATIONS")
    for i, zone in enumerate(["8-25M", "8-25M"]):
        x = 150 + 420 * i
        _text(page, x, 420, f"MUR MR{i + 1}", 11)
        _text(page, x, 330, "15M @ 300 HOR.")
        _text(page, x, 350, "15M @ 250 VERT.")
        _text(page, x, 370, zone)

    page = _sheet(doc, "S-501", "COLONNES - TABLEAU DES COLONNES")
    cells = {"NIV. 3": ["6-20M", "6-20M", "6-20M", "6-20M"], "NIV. 2": ["8-20M", "8-25M", "8-25M", "8-20M"],
             "RDC": ["8-25M", "12-25M", "8-25M", "8-25M"]}
    for c in range(4):
        _text(page, 250 + 150 * c, 120, f"C{c + 1}", 11)
    for r, (level, row) in enumerate(cells.items()):
        y = 200 + 100 * r
        _text(page, 80, y, level, 10)
        for c, vertical in enumerate(row):
            _text(page, 235 + 150 * c, y, vertical)
            _text(page, 235 + 150 * c, y + 14, "LIG. 10M @ 300")

    page = _sheet(doc, "S-601", "DALLE NIVEAU 2 - ARMATURE")
    page.draw_rect(pymupdf.Rect(80, 80, 880, 600), width=1)
    _text(page, 120, 140, "15M @ 300 INF. C.S.")
    _text(page, 520, 220, "10-20M x 3000 SUP.")
    _text(page, 300, 420, "6-15M x 2400 SUP.")
    _text(page, 680, 500, "12-15M @ 250 SUP.")

    doc.save(path)
    doc.close()


def _detail_sheet(path: Path, title: str, details: list[tuple[str, list[str]]]) -> None:
    """One shop sheet: each detail is a title with its callouts stacked above."""
    doc = pymupdf.open()
    page = doc.new_page(width=PAGE[0], height=PAGE[1])
    _text(page, PAGE[0] - 320, PAGE[1] - 40, title, 11)
    for i, (label, callouts) in enumerate(details):
        x = 80 + 270 * (i % 4)
        y = 260 + 300 * (i // 4)
        page.draw_rect(pymupdf.Rect(x - 10, y - 110, x + 200, y + 20), width=0.7)
        _text(page, x, y, label, 11)
        for k, callout in enumerate(reversed(callouts)):
            _text(page, x, y - 30 - 16 * k, callout)
    doc.save(path)
    doc.close()


def _rasterize(path: Path, dpi: int = 150) -> None:
    """Replace every page by its image: the PDF keeps its look and loses its text."""
    src = pymupdf.open(path)
    out = pymupdf.open()
    for page in src:
        pix = page.get_pixmap(dpi=dpi)
        new = out.new_page(width=page.rect.width, height=page.rect.height)
        new.insert_image(new.rect, pixmap=pix)
    src.close()
    out.save(path)
    out.close()


def build_project(root: Path, name: str = "DEMO", raster_slab: bool = False) -> Path:
    """Create `<root>/<name>/` with a plan and shop drawings.  Returns the project folder."""
    project = Path(root) / name
    da = project / "DA"
    for folder in ("Fondations", "Poutres", "Murs refends", "Colonnes", "Dalles"):
        (da / folder).mkdir(parents=True, exist_ok=True)
    _plan(project / f"L2C_PLAN_STR_{name}.pdf")

    _detail_sheet(da / "Fondations" / f"{name}_SEMELLES.pdf", "SEMELLES", [
        ("SEMELLE S1", ["8-20M A01 @ 250 INF."]),
        ("SEMELLE S2", ["10-20M A02 @ 200 INF."]),
        ("SEMELLE S3", ["6-20M A03 @ 250 INF."]),  # planted: plan asks for 8
        ("SEMELLE S4", ["12-25M A04 @ 150 INF."]),
    ])
    _detail_sheet(da / "Poutres" / f"{name}_POUTRES.pdf", "POUTRES", [
        ("POUTRE P1", ["3-25M B01 SUP.", "4-25M B02 INF.", "24-10M E01 @ 200"]),
        ("POUTRE P2", ["2-20M B03 SUP.", "3-25M B04 INF.", "20-10M E02 @ 250"]),  # planted: plan asks @ 200
        ("POUTRE P3", ["3-20M B05 SUP.", "3-20M B06 INF.", "30-10M E03 @ 150"]),
    ])
    _detail_sheet(da / "Murs refends" / f"{name}_MURS REFENDS.pdf", "MURS DE REFEND", [
        ("MUR MR1", ["15M H01 @ 300 HOR.", "15M V01 @ 250 VERT.", "8-25M Z01"]),
        ("MUR MR2", ["15M H02 @ 300 HOR.", "15M V02 @ 250 VERT.", "8-20M Z02"]),  # planted: plan asks 25M
    ])
    _detail_sheet(da / "Colonnes" / f"{name}_COLONNE-NIV-RDC@2.pdf", "COLONNES RDC @ 2", [
        ("COLONNE C1", ["8-25M V11", "14-10M L11 @ 300"]),
        ("COLONNE C2", ["12-25M V12", "14-10M L12 @ 300"]),
        ("COLONNE C3", ["6-25M V13", "14-10M L13 @ 300"]),  # planted: plan asks for 8; C4 is left out
    ])
    _detail_sheet(da / "Colonnes" / f"{name}_COLONNE-NIV-2@3.pdf", "COLONNES NIV 2 @ 3", [
        ("COLONNE C1", ["8-20M V21", "14-10M L21 @ 300"]),
        ("COLONNE C2", ["8-25M V22", "14-10M L22 @ 300"]),
        ("COLONNE C3", ["8-25M V23", "14-10M L23 @ 300"]),
        ("COLONNE C4", ["8-20M V24", "14-10M L24 @ 300"]),
        ("COLONNE C9", ["4-15M V29", "10-10M L29 @ 300"]),  # planted: not in the plan
    ])

    slab = da / "Dalles" / f"{name}_DALLE NIV 2.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=PAGE[0], height=PAGE[1])
    page.draw_rect(pymupdf.Rect(60, 60, 1000, 700), width=1)
    _text(page, 110, 135, "40-15M D01 @ 300 INF.", 10)
    _text(page, 110, 155, "36-15M D02 @ 300 INF.", 10)
    _text(page, 610, 250, "8-20M D03 x 3000 SUP.", 10)  # planted: plan asks for 10
    _text(page, 350, 500, "6-15M D04 x 2400 SUP.", 10)
    _text(page, 800, 595, "12-15M D05 @ 250 SUP.", 10)
    doc.save(slab)
    doc.close()
    if raster_slab:
        _rasterize(slab)
    return project


# --------------------------------------------------------------------------
# A second made-up project, drawn the way column plans usually are: a plan view
# per storey, columns as filled rectangles on the grid, each with a tag beside it.

GRID_NUMBERS = ["1", "2", "3", "4"]
GRID_LETTERS = ["A", "B", "C"]
GRID_PLANTED = [
    {"feuillet": "S-501", "lieu": "C-3", "attribut": "quantite", "plan": "8-25M", "atelier": "6-25M"},
    {"feuillet": "S-501", "lieu": "B-4", "attribut": "espacement_mm", "plan": "10M@300", "atelier": "10M@400"},
    {"feuillet": "S-502", "lieu": "A-4", "attribut": "diametre", "plan": "8-25M", "atelier": "8-20M"},
]


def _grid_x(number: str) -> float:
    return 260.0 + 220.0 * GRID_NUMBERS.index(number)


def _grid_y(letter: str) -> float:
    return 180.0 + 190.0 * GRID_LETTERS.index(letter)


def _column_plan(doc: pymupdf.Document, number: str, title: str, verticals: dict[str, str]) -> None:
    page = _sheet(doc, number, title)
    for n in GRID_NUMBERS:  # bubbles along the top and the bottom
        for y in (70.0, 640.0):
            page.draw_circle((_grid_x(n), y - 5), 13, width=0.7)
            _text(page, _grid_x(n) - 4, y, n, 16)
        page.draw_line((_grid_x(n), 90), (_grid_x(n), 620), width=0.3, dashes="[6 4] 0")
    for letter in GRID_LETTERS:  # bubbles down both sides
        for x in (90.0, 1010.0):
            page.draw_circle((x + 5, _grid_y(letter) - 5), 13, width=0.7)
            _text(page, x, _grid_y(letter), letter, 16)
        page.draw_line((110, _grid_y(letter)), (990, _grid_y(letter)), width=0.3, dashes="[6 4] 0")
    for letter in GRID_LETTERS:
        for n in GRID_NUMBERS:
            cx, cy = _grid_x(n), _grid_y(letter)
            page.draw_rect(pymupdf.Rect(cx - 6, cy - 9, cx + 6, cy + 9), color=(0, 0, 0), fill=(0, 0, 0))
            # The tag sits left of its column; the first of each row is placed on the right instead.
            tx = cx + 22 if n == "1" else cx - 92
            ty = cy - 6
            page.draw_rect(pymupdf.Rect(tx - 3, ty - 9, tx + 66, ty + 24), width=0.5)
            page.draw_line((tx + 66 if tx < cx else tx - 3, cy), (cx - 6 if tx < cx else cx + 6, cy), width=0.5)
            lines = ["COL. 400x600", f"ARM.: {verticals.get(letter + '-' + n, '8-25M')}", "LIG.: 10M@300 c/c",
                     "BETON: 30MPa / N"]
            for k, line in enumerate(lines):
                _text(page, tx, ty + 7.5 * k, line, 6)


def _grid_shop(path: Path, title: str, details: list[tuple[str, list[str]]]) -> None:
    doc = pymupdf.open()
    page = doc.new_page(width=PAGE[0], height=PAGE[1])
    _text(page, PAGE[0] - 320, PAGE[1] - 40, title, 11)
    for i, (names, callouts) in enumerate(details):
        x = 60 + 195 * (i % 6)
        y = 230 + 250 * (i // 6)
        page.draw_rect(pymupdf.Rect(x - 8, y - 90, x + 170, y + 18), width=0.7)
        _text(page, x, y, f"COL. {names}", 11)
        for k, callout in enumerate(reversed(callouts)):
            _text(page, x, y - 30 - 16 * k, callout)
    doc.save(path)
    doc.close()


def build_grid_project(root: Path, name: str = "GRILLE") -> Path:
    """Create `<root>/<name>/`: column plans read by grid position, with three planted discrepancies."""
    project = Path(root) / name
    (project / "DA" / "Colonnes").mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open()
    _column_plan(doc, "S-501", "COLONNES NIVEAU RDC", {"B-2": "12-25M"})
    _column_plan(doc, "S-502", "COLONNES NIVEAU 2", {"B-2": "8-30M", "C-1": "6-20M"})
    doc.save(project / f"L2C_PLAN_STR_{name}.pdf")
    doc.close()

    def details(verticals: dict[str, str], ties: dict[str, str], shared: tuple[str, str]) -> list[tuple[str, list[str]]]:
        out: list[tuple[str, list[str]]] = []
        for letter in GRID_LETTERS:
            for n in GRID_NUMBERS:
                ref = f"{letter}-{n}"
                if ref == shared[1]:
                    continue  # drawn once, with its twin
                mark = f"{len(out) + 1:02d}"
                if ref == shared[0]:
                    # one detail for two identical columns, with the totals written on it
                    out.append((f"{shared[0]}, {shared[1]}", [f"V: 16 25M 25A{mark}", f"T: 28 10M 10A{mark} @ 300"]))
                    continue
                size = verticals.get(ref, "8 25M")
                out.append((ref, [f"V: {size} {size[-3:-1]}A{mark}", f"T: 14 10M 10A{mark} @ {ties.get(ref, '300')}"]))
        return out

    colonnes = project / "DA" / "Colonnes"
    _grid_shop(colonnes / f"{name}_COLONNE-NIV-RDC@2.pdf", "COLONNES RDC @ 2",
               details({"B-2": "12 25M", "C-3": "6 25M"}, {"B-4": "400"}, ("A-2", "A-3")))
    _grid_shop(colonnes / f"{name}_COLONNE-NIV-2@3.pdf", "COLONNES NIV 2 @ 3",
               details({"B-2": "8 30M", "C-1": "6 20M", "A-4": "8 20M"}, {}, ("B-3", "B-4")))
    return project


def write_known_list(path: Path, rows: list[dict]) -> Path:
    """Spreadsheet of documented non-conformities, in the layout `evaluate.load_known` reads."""
    import openpyxl

    book = openpyxl.Workbook()
    sheet = book.active
    sheet.append(["Feuillet", "Emplacement", "Plan", "Dessin d'atelier"])
    for row in rows:
        sheet.append([row["feuillet"], row["lieu"], row["plan"], row["atelier"]])
    book.save(path)
    return path
