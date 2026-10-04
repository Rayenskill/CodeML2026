"""Element labels (C-12, S3, MR-2...), building levels and element types."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable, Optional

from ..config import AUTRE, LABEL_STOP, SERIES_TYPES, TYPE_KEYWORDS, TYPE_PREFIXES

_LABEL_RE = re.compile(
    r"(?<![A-Za-z0-9])(?P<pre>[A-Z]{1,4})(?P<sep>\s?[-.]\s?|\s)?(?P<num>\d{1,3})(?P<suf>[A-Za-z])?(?![A-Za-z0-9])"
)
_SHEET_RE = re.compile(r"(?<![A-Za-z0-9])(?P<disc>[A-Z]{1,2})\s?-\s?(?P<num>\d{3})(?P<suf>[A-Za-z])?(?![A-Za-z0-9])")
_SIZE_TOKEN = re.compile(r"^(10|15|20|25|30|35|45|55)M$")


def strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")


def norm_text(text: str) -> str:
    return strip_accents(text).upper()


@dataclass
class LabelMatch:
    start: int
    end: int
    label: str  # normalised, e.g. "C12", "MR1A"
    prefix: str
    known: bool  # prefix is a usual one for the page's element type


def normalize_label(prefix: str, num: str, suffix: str | None = None) -> str:
    return f"{prefix.upper()}{int(num)}{(suffix or '').upper()}"


_GRID_REF_RE = re.compile(
    r"(?<![A-Za-z0-9.\-/])(?P<letter>[A-Z]{1,2}(?:\.\d)?)'?\s?[-/]\s?(?P<number>\d{1,2}(?:\.\d)?)(?![0-9A-Za-z\"'./])"
)


def find_grid_refs(text: str) -> list[LabelMatch]:
    """Grid crossings used as element names on shop drawings: "B-3", "B/3", "A.1-12"."""
    out = []
    for m in _GRID_REF_RE.finditer(text):
        if text[: m.start()].rstrip().endswith("/"):
            continue  # concrete exposure class ("25MPa / C-1")
        number = m.group("number")
        number = number if "." in number else str(int(number))
        out.append(LabelMatch(m.start(), m.end(), f"{m.group('letter')}-{number}", m.group("letter"), True))
    return out


def find_labels(text: str, type_element: str, extra: Iterable[str] = (), grid_refs: bool = False) -> list[LabelMatch]:
    """Find element labels in a line.

    `extra` holds labels learnt from the documents themselves (labels found on
    both the plan and the shop drawings), accepted whatever their prefix.
    With `grid_refs`, grid crossings ("B-3") count as element names too.
    """
    known_prefixes = TYPE_PREFIXES.get(type_element, frozenset())
    extra = set(extra)
    out: list[LabelMatch] = find_grid_refs(text) if grid_refs else []
    taken = [(m.start, m.end) for m in out]
    for m in _LABEL_RE.finditer(text):
        pre, num, sep = m.group("pre"), m.group("num"), m.group("sep") or ""
        if len(num) == 3 and len(pre) <= 2 and "-" in sep:
            continue  # sheet reference such as S-501
        if text[: m.start()].rstrip().endswith("/"):
            continue  # concrete exposure class ("35MPa / C-1"), not an element
        if any(m.start() < end and start < m.end() for start, end in taken):
            continue  # already read as a grid crossing
        label = normalize_label(pre, num, m.group("suf"))
        known = pre in known_prefixes
        if not known and label not in extra:
            continue
        if sep.isspace() and not known:
            continue
        if _SIZE_TOKEN.match(text[m.start(): m.end()].replace(" ", "")):
            continue
        out.append(LabelMatch(m.start(), m.end(), label, pre, known))
    return out


def label_candidates(text: str) -> list[str]:
    """All label-shaped tokens of a line, for vocabulary discovery."""
    out = []
    for m in _LABEL_RE.finditer(text):
        pre, sep = m.group("pre"), m.group("sep") or ""
        if pre in LABEL_STOP or sep.isspace() or text[: m.start()].rstrip().endswith("/"):
            continue
        if len(m.group("num")) == 3 and len(pre) <= 2 and "-" in sep:
            continue
        out.append(normalize_label(pre, m.group("num"), m.group("suf")))
    return out


# ----------------------------------------------------------------- sheets

def find_sheet_ids(text: str) -> list[tuple[int, int, str]]:
    """Sheet numbers (S-501, S-601A...) present in a line, normalised."""
    return [
        (m.start(), m.end(), f"{m.group('disc')}-{m.group('num')}{(m.group('suf') or '').upper()}")
        for m in _SHEET_RE.finditer(text)
    ]


def type_from_sheet(feuillet: str) -> Optional[str]:
    m = re.match(r"^S-(\d)\d{2}", feuillet)
    return SERIES_TYPES.get(int(m.group(1))) if m else None


def type_from_text(*texts: str) -> Optional[str]:
    """Element type from folder / file / title keywords; the first text that hits wins."""
    for text in texts:
        t = norm_text(text or "")
        for type_element, keywords in TYPE_KEYWORDS:
            if any(k in t for k in keywords):
                return type_element
    return None


# ----------------------------------------------------------------- levels

_LEVEL_TOKEN = (
    r"(?:"
    r"(?P<{p}niv>NIV(?:EAUX?|\.)?\s?[-.]?\s?(?P<{p}nivn>\d{{1,2}}))"
    r"|(?P<{p}ss>(?:SS|SOUS[- ]?SOL)\s?[-.]?\s?(?P<{p}ssn>\d)?)"
    r"|(?P<{p}rdc>RDC|R\.D\.C\.?|REZ[- ]DE[- ]CHAUSSEE)"
    r"|(?P<{p}toit>TOIT(?:URE)?|APPENTIS)"
    r"|(?P<{p}tref>TREFOND)"
    r"|(?P<{p}fdn>FDN|FND|FONDATIONS?)"
    r"|(?P<{p}mezz>MEZZ(?:ANINE)?)"
    r"|(?P<{p}eta>(?P<{p}etan>\d{{1,2}})\s?(?:E|ER|IEME|EME)\s+(?:ETAGE|NIVEAU))"
    r")"
)
_LEVEL_RE = re.compile(r"(?<![A-Z0-9])" + _LEVEL_TOKEN.format(p="a") + r"(?![A-Z0-9])")
_RANGE_RE = re.compile(
    r"(?<![A-Z0-9])" + _LEVEL_TOKEN.format(p="a") + r"\s*(?:@|\bA\b|\bAU\b|-|/)\s*(?:" + _LEVEL_TOKEN.format(p="b")
    + r"|(?P<bnum>\d{1,2}))(?![A-Z0-9])"
)


def _canon(m: re.Match[str], p: str) -> Optional[str]:
    g = m.groupdict()
    if g.get(f"{p}niv"):
        return str(int(g[f"{p}nivn"]))
    if g.get(f"{p}ss"):
        return f"SS{g[f'{p}ssn'] or 1}"
    if g.get(f"{p}rdc"):
        return "RDC"
    if g.get(f"{p}toit"):
        return "TOIT"
    if g.get(f"{p}tref"):
        return "TREFOND"
    if g.get(f"{p}fdn"):
        return "FDN"
    if g.get(f"{p}mezz"):
        return "MEZZ"
    if g.get(f"{p}eta"):
        return str(int(g[f"{p}etan"]))
    return None


def find_levels(text: str) -> tuple[str, ...]:
    """Canonical building levels named in a title or a file name.

    "COLONNE-NIV-2@3" -> ("2", "3");  "DALLE NIV RDC" -> ("RDC",);
    "RADIERS" -> ().  A range keeps (from, to) in that order.
    """
    t = norm_text(text).replace("_", " ")
    m = _RANGE_RE.search(t)
    if m:
        a = _canon(m, "a")
        b = m.group("bnum") and str(int(m.group("bnum"))) or _canon(m, "b")
        if a and b:
            return (a, b)
    seen: list[str] = []
    for m in _LEVEL_RE.finditer(t):
        c = _canon(m, "a")
        if c and c not in seen:
            seen.append(c)
    return tuple(seen)


def level_tokens(text: str) -> list[tuple[int, int, str]]:
    """Level names inside a drawing line, used to tag schedule rows."""
    t = norm_text(text)
    return [(m.start(), m.end(), c) for m in _LEVEL_RE.finditer(t) if (c := _canon(m, "a"))]


def resolve_type(feuillet: str, *texts: str) -> str:
    return type_from_sheet(feuillet) or type_from_text(*texts) or AUTRE
