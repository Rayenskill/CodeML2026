"""Data models.

`Record` / `Armature` are the public JSON contract (Appendix A of the challenge
rules) and are validated with pydantic.  Everything else is internal working
state that carries what the schema has no room for (bounding boxes, confidence,
level, raw text) and ends up in the side-car "details" file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict

BBox = tuple[float, float, float, float]

SOURCE_PLAN = "plan"
SOURCE_SHOP = "atelier"

# Statuses required by the rules (section 5, step 4).
CONFORME = "conforme"
NON_CONFORME = "non_conforme"
MANQUANT = "manquant_atelier"
AJOUTE = "ajoute_atelier"
STATUSES = (CONFORME, NON_CONFORME, MANQUANT, AJOUTE)


class Armature(BaseModel):
    """One reinforcing-bar entry, exactly as in Appendix A."""

    model_config = ConfigDict(extra="forbid")

    repere: Optional[str] = None
    diametre: Optional[str] = None
    quantite: Optional[int] = None
    espacement_mm: Optional[int] = None
    longueur_mm: Optional[int] = None


class Record(BaseModel):
    """One identified piece of information, exactly as in Appendix A.

    x, y are PDF points from the top-left corner of the displayed page and
    point at the centre of the annotation.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    source: Literal["plan", "atelier"]
    fichier: str
    feuillet: str
    page: int
    x: float
    y: float
    type_element: str
    element: str
    armature: list[Armature]


@dataclass
class TextLine:
    """A line of text with its box in displayed page coordinates (top-left origin)."""

    text: str
    bbox: BBox
    size: float = 8.0
    dx: float = 1.0  # reading direction on the displayed page
    dy: float = 0.0
    origin: str = "text"  # "text" (vector text layer) or "ocr"
    conf: float = 1.0

    @property
    def vertical(self) -> bool:
        return abs(self.dy) > abs(self.dx)

    def span_bbox(self, start: int, end: int) -> BBox:
        """Box of characters [start, end), interpolated along the reading direction."""
        n = max(len(self.text), 1)
        fa, fb = max(0.0, start / n), min(1.0, end / n)
        x0, y0, x1, y1 = self.bbox
        if not self.vertical:
            if self.dx >= 0:
                return (x0 + (x1 - x0) * fa, y0, x0 + (x1 - x0) * fb, y1)
            return (x1 - (x1 - x0) * fb, y0, x1 - (x1 - x0) * fa, y1)
        if self.dy < 0:  # reads bottom to top
            return (x0, y1 - (y1 - y0) * fb, x1, y1 - (y1 - y0) * fa)
        return (x0, y0 + (y1 - y0) * fa, x1, y0 + (y1 - y0) * fb)


@dataclass
class Bar:
    """A parsed rebar callout (one `armature` entry plus working data)."""

    diametre: Optional[str] = None
    quantite: Optional[int] = None
    espacement_mm: Optional[int] = None
    longueur_mm: Optional[int] = None
    repere: Optional[str] = None
    role: Optional[str] = None  # sup / inf / vert / hor / ligature / goujon
    raw: str = ""
    bbox: BBox = (0.0, 0.0, 0.0, 0.0)
    conf: float = 1.0
    level: Optional[str] = None  # schedule row level, when the callout sits in a level row
    bordereau: bool = False  # row of a bar list rather than a placing callout

    def to_armature(self) -> Armature:
        return Armature(
            repere=self.repere,
            diametre=self.diametre,
            quantite=self.quantite,
            espacement_mm=self.espacement_mm,
            longueur_mm=self.longueur_mm,
        )

    def describe(self) -> str:
        """Compact engineering notation, e.g. `8-25M @ 300 x 3600`."""
        s = f"{self.quantite}-" if self.quantite is not None else ""
        s += self.diametre or "?"
        if self.espacement_mm is not None:
            s += f" @ {self.espacement_mm}"
        if self.longueur_mm is not None:
            s += f" x {self.longueur_mm}"
        return s


@dataclass
class Element:
    """A structural element (or an unlabeled callout group) found on one page."""

    source: str
    fichier: str
    feuillet: str
    page: int  # 1-based
    type_element: str
    element: str
    x: float
    y: float
    bars: list[Bar] = field(default_factory=list)
    labeled: bool = False
    levels: tuple[str, ...] = ()  # canonical levels this element instance covers
    conf: float = 1.0
    page_size: tuple[float, float] = (0.0, 0.0)
    path: str = ""  # absolute path of the source PDF (not exported)
    id: str = ""
    label: Optional[str] = None  # normalised label used for matching (C12); None when unlabeled
    bordereau: bool = False  # bar-list rows of a shop drawing: exported, never compared
    multiplicity: int = 1  # number of elements sharing this detail ("B-12, B-13" -> 2)
    ocr: bool = False  # read by OCR rather than from a text layer
    grid: Optional[tuple[float, float, float, float]] = None  # place on the building grid (see GridAxes.coords)
    grid_ref: Optional[str] = None  # the same place as an engineer writes it: "B-12", "B-C/11-12"

    @property
    def key(self) -> str:
        return self.element

    def to_record(self) -> Record:
        return Record(
            id=self.id,
            source=self.source,  # type: ignore[arg-type]
            fichier=self.fichier,
            feuillet=self.feuillet,
            page=self.page,
            x=round(self.x, 1),
            y=round(self.y, 1),
            type_element=self.type_element,
            element=self.element,
            armature=[b.to_armature() for b in self.bars],
        )


@dataclass
class Ecart:
    """One attribute discrepancy between a plan bar and a shop-drawing bar."""

    attribut: str  # quantite / diametre / espacement_mm / longueur_mm / absence
    plan: object
    atelier: object
    gravite: str  # critique / majeur / mineur
    message: str
    plan_bbox: Optional[BBox] = None
    atelier_bbox: Optional[BBox] = None
    plan_bar: Optional[Bar] = None  # the plan bar at fault, kept for follow-up checks (not exported)


@dataclass
class Result:
    """Classification of one element (plan side, shop side, or both)."""

    statut: str
    type_element: str
    element: str
    feuillet: str  # plan sheet the result is reported under
    plan: Optional[Element] = None
    atelier: Optional[Element] = None
    ecarts: list[Ecart] = field(default_factory=list)
    confiance: float = 1.0
    methode: str = "repere"  # repere (label match) / signature (content match)
    note: str = ""
    id: str = ""
    verifiees: int = 0  # plan bars confirmed by the shop drawing in this pair

    @property
    def a_valider(self) -> bool:
        return self.confiance < 0.6

    @property
    def gravite(self) -> str:
        order = {"critique": 0, "majeur": 1, "mineur": 2}
        if not self.ecarts:
            return "mineur"
        return min((e.gravite for e in self.ecarts), key=lambda g: order.get(g, 3))
