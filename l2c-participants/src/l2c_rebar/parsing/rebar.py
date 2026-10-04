"""Grammar for rebar callouts with Canadian metric bar sizes (10M ... 55M).

Forms recognised on structural plans:

    8-25M              quantity - size
    12(6)-15M          quantity (of which 6 in the column band) - size
    8 x 25M, (8) 25M
    15M @ 300          size at spacing in mm (also "@300mm c/c", "à 300", "A 300 C/C")
    15M @ 12"          size at spacing in inches, converted to mm
    8-20M x 3000       length after "x" (also "LG 3000", "L=3000")

Forms recognised on shop drawings (detailing-software style):

    24 15M 15A12 @ 12" quantity, size, bar mark, spacing
    2x4 25M 25A301     two sets of four bars: quantity 8
    24 15M 20-6        quantity, size, straight-bar length in feet-inches
    8 25M 3600         quantity, size, straight-bar length in mm
    12-15M B12 @ 300   a mark between size and spacing
    5-20M2400          straight-bar mark with the length glued to the size

Lengths and spacings are always returned in millimetres.  Text coming from OCR
is normalised first (`normalize_ocr`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

SIZES = ("10", "15", "20", "25", "30", "35", "45", "55")
_SIZE = r"(?:10|15|20|25|30|35|45|55)"
MM_PER_INCH = 25.4

_BAR_RE = re.compile(
    r"(?<![0-9A-Za-z.,/])"
    r"(?:"
    r"(?P<q_dash>\d{1,3})\s*(?:\(\s*(?P<sub>\d{1,3})\s*\)\s*)?[-–—xX×]\s*"  # 8-25M / 12(6)-15M / 8 x 25M
    r"|\(\s*(?P<q_par>\d{1,3})\s*\)\s*"  # (8) 25M
    r"|(?:(?P<mult>\d{1,2})\s?[xX×]\s?)?(?P<q_sp>\d{1,3})\s+"  # 8 25M, 2x4 25M (two sets of four)
    r")?"
    rf"(?P<dia>{_SIZE})\s?M"
    r"(?P<glued>\d{3,5})?"  # 20M2400
    r"(?![A-Za-z0-9])"
)

_INCH_MARK = r"(?:\"|''|”|″)"
_SPACING_IN_RE = re.compile(
    rf"(?:@|\bAT\b|[àÀ])\s*(?P<inch>\d{{1,2}})(?:\s(?P<num>\d)/(?P<den>\d{{1,2}}))?\s?{_INCH_MARK}", re.IGNORECASE
)
_SPACING_RE = re.compile(
    r"(?:@|\bAT\b|[àÀ]|\bA\b|\bAUX\b)\s*(?P<esp>\d{2,4})(?![\d.,]*\s*[xX×])"
    r"|(?<![\d.,])(?P<esp_cc>\d{2,4})\s*(?:mm\s*)?(?:c\s?/\s?c|c\.\s?c\.?|c\.?\s?[àa]\.?\s?c\.?|o\.\s?c\.?|e\.\s?e\.)",
    re.IGNORECASE,
)

_LENGTH_RE = re.compile(
    r"[xX×]\s*(?P<len>\d{3,5})(?![\d.,]|\s*[xX×])"
    r"|\b(?:LG|LONG|LONGUEUR|L)\s?\.?\s?[=:]?\s*(?P<len_lg>\d{3,5})\b"
    r"|(?<![\d.,])(?P<len_post>\d{3,5})\s*(?:mm\s*)?(?:LG|LONG)\b",
    re.IGNORECASE,
)
# Right after "quantity size": a straight bar named by its length.
_LENGTH_FT_IN_RE = re.compile(r"^\s+(?P<ft>\d{1,2})-(?P<inch>\d{1,2})(?![\dA-Za-z'\"/.-])")
_LENGTH_MM_RE = re.compile(r"^\s+(?P<mm>\d{4,5})(?![\dA-Za-z'\"/.-])")
# ... or a bent bar named by a three-digit mark ("8 25M 301").
_NUMERIC_MARK_RE = re.compile(r"^\s+(?P<mark>\d{3})(?![\dA-Za-z'\"/.@-])")

_MARK_RE = re.compile(
    r"(?<![A-Za-z0-9])(?P<mark>"
    r"[A-Z]{1,3}-?\d{1,4}[A-Z]?(?:-\d{1,3})?"  # B12, V-101, C12-1
    r"|\d{1,3}(?:[A-Z]{1,3}\d{1,4}){1,2}[A-Z]{0,2}(?:-\d{1,3})?"  # 15A12, 20A01B2, 4B03-2
    r")(?![A-Za-z0-9])"
)

# Order matters: the first matching role wins.
_ROLE_RES: list[tuple[str, re.Pattern[str]]] = [
    ("ligature", re.compile(r"\b(LIG|LIGATURES?|[ÉE]TRIERS?|[ÉE]TR|CADRES?|[ÉE]PINGLES?|TIES?|STIRRUPS?|SPIRALES?)\b", re.I)),
    ("goujon", re.compile(r"\b(GOUJONS?|DOWELS?|ATTENTES?)\b", re.I)),
    ("sup", re.compile(r"\b(SUP|SUP[ÉE]RIEURE?S?|HAUT|TOP|NAPPE\s+SUP)\b", re.I)),
    ("inf", re.compile(r"\b(INF|INF[ÉE]RIEURE?S?|BAS|BOT|BOTTOM)\b", re.I)),
    ("vert", re.compile(r"\b(VERT|VERTICALE?S?|VERTICAUX)\b", re.I)),
    ("hor", re.compile(r"\b(HOR|HORIZ|HORIZONTALE?S?|HORIZONTAUX)\b", re.I)),
]
# One-letter placement tags written before a callout ("T: 24 15M ...").
_TAG_RE = re.compile(r"(?:^|[\s(])(?P<tag>[A-Z]{1,2})\s?:\s*$")
_TAG_ROLES = {"T": "sup", "TT": "sup", "B": "inf", "BB": "inf", "V": "vert", "H": "hor"}

_MARK_STOP = {"T1", "T2", "B1", "B2", "E1", "E2"}
_MARK_STOP_PREFIX = re.compile(r"^(NIV|SS|LG|X\d{3}|L\d{3}|TYP|DET|REV)")
# A number followed by a space and a size is a quantity ("24 15M") unless the
# number belongs to what precedes it ("@ 300 15M", "NIV 2 15M").
_NOT_QTY_BEFORE = re.compile(
    r"(?:[@xX×=àÀ/-]|\b(?:AT|A|NIV|NIVEAUX?|TYPE|DET|D[ÉE]TAIL|COUPE|AXE|NO|REV|PAGE|ZONE|LIT))\.?\s*$"
)
_MAX_TAIL = 48
_MAX_HEAD = 28


@dataclass
class BarMatch:
    start: int
    end: int
    diametre: str
    quantite: Optional[int] = None
    espacement_mm: Optional[int] = None
    longueur_mm: Optional[int] = None
    repere: Optional[str] = None
    role: Optional[str] = None
    raw: str = ""
    conf: float = 1.0
    space_qty: bool = False  # quantity separated from the size by a plain space
    sub_qty: Optional[int] = None  # the "(6)" of 12(6)-15M

    @property
    def informative(self) -> bool:
        """False when the match only states a diameter (nothing to reconcile)."""
        return any(v is not None for v in (self.quantite, self.espacement_mm, self.longueur_mm))


def normalize_ocr(text: str) -> str:
    """Fix the usual OCR confusions inside rebar callouts, and nothing else."""
    t = text.replace("©", "@").replace("＠", "@").replace("—", "-").replace("–", "-")
    after_m = r"(?=\s?M(?:[eEaQ&]?\s?\d|[^A-Za-z]|$))"  # the M of a bar size, not of a word
    t = re.sub(r"(?<![A-Za-z0-9])[Il|](?=[05]\s?M(?:[eEaQ&]?\s?\d|[^A-Za-z]|$))", "1", t)  # I5M -> 15M
    t = re.sub(r"(?<=[1-5])[Oo]" + after_m, "0", t)  # 2OM -> 20M
    t = re.sub(r"(?<=[1-5])[Ss]" + after_m, "5", t)  # 1SM -> 15M
    t = re.sub(rf"(?<={_SIZE}M)\s?[eEaQ&]\s?(?=\d{{1,4}}(?!\d))", " @ ", t)  # 15Me300 -> 15M @ 300
    t = re.sub(rf"(?<=\d)\s?[~_=]\s?(?={_SIZE}M(?![A-Za-z]))", "-", t)  # 8~25M -> 8-25M
    return t


def _int(value: Optional[str]) -> Optional[int]:
    return int(value) if value else None


def _first_role(text: str) -> Optional[str]:
    best: tuple[int, str] | None = None
    for role, rx in _ROLE_RES:
        m = rx.search(text)
        if m and (best is None or m.start() < best[0]):
            best = (m.start(), role)
    return best[1] if best else None


def _spacing(tail: str) -> tuple[Optional[int], tuple[int, int] | None]:
    """Spacing in mm found in the text after a size, with the span it occupies."""
    m = _SPACING_IN_RE.search(tail)
    if m:
        inches = int(m.group("inch")) + (int(m.group("num")) / int(m.group("den")) if m.group("den") not in (None, "0") and m.group("num") else 0)
        if 2 <= inches <= 60:
            return round(inches * MM_PER_INCH), m.span()
    m = _SPACING_RE.search(tail)
    if m:
        esp = _int(m.group("esp") or m.group("esp_cc"))
        if esp is not None and 40 <= esp <= 1500:
            return esp, m.span()
    return None, None


def find_bars(text: str, ocr: bool = False) -> list[BarMatch]:
    """Return every rebar callout found in one line of text."""
    if ocr:
        text = normalize_ocr(text)
    matches = list(_BAR_RE.finditer(text))
    # Where each callout really starts: a space-separated number is dropped when
    # it reads as the tail of the previous callout rather than as a quantity.
    starts, qtys, spaced = [], [], []
    for m in matches:
        qty = _int(m.group("q_dash") or m.group("q_par") or m.group("q_sp"))
        if qty and m.group("mult"):
            qty *= int(m.group("mult"))
        is_space = m.group("q_sp") is not None
        start = m.start()
        if is_space and _NOT_QTY_BEFORE.search(text[:start]):
            qty, is_space, start = None, False, m.start("dia")
        starts.append(start)
        qtys.append(qty or None)
        spaced.append(is_space)

    out: list[BarMatch] = []
    for i, m in enumerate(matches):
        nxt = starts[i + 1] if i + 1 < len(matches) else len(text)
        prev_end = matches[i - 1].end() if i else 0
        tail = text[m.end(): min(nxt, m.end() + _MAX_TAIL)]
        head = text[max(prev_end, starts[i] - _MAX_HEAD): starts[i]]
        bar = BarMatch(start=starts[i], end=m.end(), diametre=f"{m.group('dia')}M", quantite=qtys[i],
                       space_qty=spaced[i], sub_qty=_int(m.group("sub")))

        used: list[tuple[int, int]] = []  # tail spans already explained (spacing, length)
        bar.espacement_mm, span = _spacing(tail)
        if span:
            bar.end = max(bar.end, m.end() + span[1])
            used.append(span)

        if m.group("glued"):
            bar.longueur_mm = int(m.group("glued"))
            bar.repere = f"{bar.diametre}{m.group('glued')}"
        else:
            length, lm = None, None
            if bar.space_qty:
                lm = _LENGTH_FT_IN_RE.match(tail)
                if lm and int(lm.group("inch")) < 12:
                    length = round((int(lm.group("ft")) * 12 + int(lm.group("inch"))) * MM_PER_INCH)
                else:
                    lm = _LENGTH_MM_RE.match(tail)
                    length = int(lm.group("mm")) if lm else None
            if length is None:
                lm = _LENGTH_RE.search(tail)
                length = _int(lm.group("len") or lm.group("len_lg") or lm.group("len_post")) if lm else None
            if lm and length is not None and 200 <= length <= 20000:
                bar.longueur_mm = length
                bar.end = max(bar.end, m.end() + lm.end())
                used.append(lm.span())

        if bar.repere is None and bar.space_qty and bar.longueur_mm is None:
            nm = _NUMERIC_MARK_RE.match(tail)
            if nm:
                bar.repere = nm.group("mark")
                bar.end = max(bar.end, m.end() + nm.end())
        if bar.repere is None:
            for mm in _MARK_RE.finditer(tail[:24]):
                mark = mm.group("mark")
                overlaps = any(mm.start() < e and s < mm.end() for s, e in used)
                if not overlaps and mark not in _MARK_STOP and not _MARK_STOP_PREFIX.match(mark):
                    bar.repere = mark
                    bar.end = max(bar.end, m.end() + mm.end())
                    break

        tag = _TAG_RE.search(head)
        bar.role = _first_role(tail) or _first_role(head) or (_TAG_ROLES.get(tag.group("tag")) if tag else None)
        bar.raw = text[bar.start: bar.end].strip()

        if m.group("q_dash") or m.group("q_par"):
            bar.conf = 0.95
        elif bar.space_qty:
            # "24 15M" alone is weak; followed by a mark, a length or a spacing it is a shop callout.
            bar.conf = 0.9 if (bar.repere or bar.longueur_mm or bar.espacement_mm) else 0.6
        elif bar.espacement_mm is not None:
            bar.conf = 0.9
        else:
            bar.conf = 0.4
        out.append(bar)
    return out
