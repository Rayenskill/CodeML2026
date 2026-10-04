"""Score a run against a list of documented non-conformities.

The list is a spreadsheet with one row per known discrepancy:

    plan sheet | location on the grid | what the plan says | what the shop drawing says

(for instance: S-502 | B-12 | 6-25M | 4-25M).  The score says, row by row, how far
the pipeline got: was the sheet read, was the plan annotation extracted at that
place, was a discrepancy reported there, and does it state the same values.

Only counts and stage flags are printed, never the content of the list.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .extract.grid import GridAxes, find_grid_axes
from .models import CONFORME, Element, Result
from .parsing.elements import find_sheet_ids, type_from_sheet
from .parsing.rebar import find_bars

_GRID_TOKEN = re.compile(r"(?<![A-Za-z0-9.])([A-Z]{1,2}(?:\.\d)?|\d{1,2}(?:\.\d)?)(?![A-Za-z0-9])")
_COUNT_ONLY = re.compile(r"^\s*(\d{1,3})\s*(?:\(\s*(\d{1,3})\s*\))?\s*$")


@dataclass
class Known:
    row: int
    feuillet: str
    letters: set[str]
    numbers: set[str]
    plan: str
    atelier: str


@dataclass
class Score:
    row: int
    type_element: str
    sheet_read: bool = False
    plan_value_on_sheet: bool = False  # the plan annotation was extracted somewhere on the sheet
    plan_value_at_location: bool = False  # ... and at the stated grid location
    reported_at_location: bool = False  # a discrepancy or a missing element is reported there
    same_values: bool = False  # ... stating the same plan and shop values
    notes: list[str] = field(default_factory=list)

    @property
    def found(self) -> bool:
        return self.reported_at_location


def _signature(text: str) -> tuple | None:
    """(diameter, quantity, spacing) of a value written in plan notation; counts alone give (None, n, None)."""
    bars = find_bars(str(text))
    if bars:
        b = bars[0]
        return (b.diametre, b.quantite, b.espacement_mm)
    m = _COUNT_ONLY.match(str(text))
    return (None, int(m.group(1)), None) if m else None


def _same(sig: tuple | None, bar) -> bool:
    if sig is None:
        return False
    dia, qty, esp = sig
    return (dia is None or bar.diametre == dia) and (qty is None or bar.quantite == qty) and \
        (esp is None or (bar.espacement_mm is not None and abs(bar.espacement_mm - esp) <= 6))


def load_known(path: Path) -> list[Known]:
    """Read the spreadsheet: first row is the header, columns are sheet, location, plan, shop."""
    import openpyxl

    sheet = openpyxl.load_workbook(path, data_only=True).worksheets[0]
    known = []
    for i, row in enumerate(sheet.iter_rows(min_row=2, values_only=True), start=2):
        cells = [("" if c is None else str(c).strip()) for c in row[:4]] + [""] * 4
        ids = find_sheet_ids(cells[0].upper())
        if not ids:
            continue
        tokens = _GRID_TOKEN.findall(cells[1].upper())
        known.append(Known(row=i, feuillet=ids[0][2], letters={t for t in tokens if t[0].isalpha()},
                           numbers={t for t in tokens if t[0].isdigit()}, plan=cells[2], atelier=cells[3]))
    return known


def _place(axes: GridAxes, x: float, y: float, reach: int = 1) -> tuple[set[str], set[str]]:
    """Grid labels around a point: the nearest line of each direction and its neighbours."""
    def around(lines: list[tuple[str, float]], pos: float) -> set[str]:
        ordered = sorted(lines, key=lambda it: it[1])
        if not ordered:
            return set()
        i = min(range(len(ordered)), key=lambda k: abs(ordered[k][1] - pos))
        return {ordered[k][0] for k in range(max(0, i - reach), min(len(ordered), i + reach + 1))}

    across, down = around(axes.vertical, x), around(axes.horizontal, y)
    return (across, down) if axes.letters_are_vertical else (down, across)


def _at(known: Known, el: Element, axes: GridAxes | None) -> bool:
    if el.label and "/" in el.label:  # between labelled lines ("B-C/12"): the list may name the line "B.1"
        letter, number = el.label.split("/", 1)

        def fits(tokens: set[str], part: str) -> bool:
            if "-" not in part:  # on the line itself
                return part in tokens
            # between two lines: "B.1" sits after B, so it is compared with the lower line only
            return any(t.split(".")[0] == part.split("-")[0] for t in tokens)

        return (not known.letters or fits(known.letters, letter)) and (not known.numbers or fits(known.numbers, number))
    if el.label and "-" in el.label:  # already named by its grid crossing
        letter, number = el.label.split("-", 1)
        return (not known.letters or letter in known.letters) and (not known.numbers or number in known.numbers)
    if axes is None or not axes.usable:
        return False
    letters, numbers = _place(axes, el.x, el.y)
    return bool((not known.letters or known.letters & letters) and (not known.numbers or known.numbers & numbers))


def score_run(known: list[Known], plan_elements: list[Element], results: list[Result],
              axes_by_sheet: dict[str, GridAxes]) -> list[Score]:
    scores = []
    for k in known:
        score = Score(k.row, type_from_sheet(k.feuillet) or "autre")
        axes = axes_by_sheet.get(k.feuillet)
        on_sheet = [e for e in plan_elements if e.feuillet == k.feuillet]
        score.sheet_read = bool(on_sheet)
        plan_sig, shop_sig = _signature(k.plan), _signature(k.atelier)
        holders = [e for e in on_sheet if any(_same(plan_sig, b) for b in e.bars)]
        score.plan_value_on_sheet = bool(holders)
        score.plan_value_at_location = any(_at(k, e, axes) for e in holders)
        for r in results:
            if r.statut == CONFORME or r.feuillet != k.feuillet or r.plan is None or not _at(k, r.plan, axes):
                continue
            if not any(_same(plan_sig, b) for b in r.plan.bars):
                continue
            score.reported_at_location = True
            if r.atelier is not None and any(_same(shop_sig, b) for b in r.atelier.bars):
                score.same_values = True
        scores.append(score)
    return scores


def axes_of_plan(plan_pages) -> dict[str, GridAxes]:
    return {page.feuillet: find_grid_axes(page.lines) for page in plan_pages}


def summarize_scores(scores: list[Score]) -> dict[str, int]:
    return {
        "connues": len(scores),
        "feuillet_lu": sum(s.sheet_read for s in scores),
        "valeur_plan_extraite": sum(s.plan_value_on_sheet for s in scores),
        "valeur_plan_au_bon_endroit": sum(s.plan_value_at_location for s in scores),
        "signalees": sum(s.reported_at_location for s in scores),
        "memes_valeurs": sum(s.same_values for s in scores),
    }
