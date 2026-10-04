"""Strategy 10: identifier leak guard + code-based patient linking.

* Identifier zones (name, husband's name, CIN, phone, address) are masked on the registered image before
  any recognition (extract_zonal.mask_identifiers); this module adds a *leak scanner* on every output.
* Linking: the midwife writes a random code on the registry ("N° de la fiche", e.g. 2026-823-001).
  Candidates tolerate OCR confusions (weighted edit distance), are ranked with non-identifying
  consistency checks (DDR, expected delivery date, facility), and the midwife always decides:
  [Patiente 1] [Patiente 2] [Aucune, créer] [Je ne sais pas]. Internal ids are random UUIDs.
"""
from __future__ import annotations

import re
import uuid

CIN_RE = re.compile(r"\b[A-Z]{1,2}\s?\d{5,7}\b")
PHONE_RE = re.compile(r"(?:\+212|0)\s?[5-7](?:[\s.-]?\d{2}){4}\b")
ADDRESS_RE = re.compile(r"\b(rue|avenue|av\.|bd|boulevard|hay|douar|quartier|lot|n°\s?\d+|حي|زنقة|شارع)\b", re.I)
NAME_LABELS = ("nom", "prénom", "prenom", "mari", "conjoint", "parturiente", "patiente", "اسم")


PLACE_FIELDS = ("etablissement", "region", "province", "lieu")     # facility names are not personal addresses


def scan_text(s: str, key: str = "") -> list[str]:
    if not s:
        return []
    hits = []
    if CIN_RE.search(s.upper()):
        hits.append("CIN")
    if PHONE_RE.search(s):
        hits.append("PHONE")
    if ADDRESS_RE.search(s) and not any(p in key for p in PLACE_FIELDS):
        hits.append("ADDRESS")
    return hits


def leak_guard(page: dict, known_names: set[str] | None = None) -> list[dict]:
    """Drop any field value that looks like a direct identifier; return incidents (never the value)."""
    incidents = []
    for f in page.get("fields", []):
        v = f.get("value")
        if not isinstance(v, str):
            continue
        hits = scan_text(v, f["key"])
        if known_names and any(n.lower() in v.lower() for n in known_names):
            hits.append("NAME")
        if any(lbl in f["key"] for lbl in ("parturiente", "patiente", "cin", "adresse", "telephone", "nom_du_mari")):
            hits.append("IDENTIFIER_FIELD")
        if hits:
            incidents.append(dict(key=f["key"], kinds=sorted(set(hits))))
            # an identifier zone is simply never kept; an identifier-like value in another field is removed and asked
            # again (the midwife writes what is needed without the identifier), never silently turned into "empty"
            ident_zone = "IDENTIFIER_FIELD" in hits
            f["value"], f["raw"], f["status"] = None, None, "NON_FOURNI" if ident_zone else "À_RÉVISER"
            f.pop("evidence", None)
            f["redacted"] = True
    return incidents


# ------------------------------------------------------------------------------------------ linking
CONFUSABLE = {frozenset(p) for p in ["01", "17", "38", "56", "69", "49", "0O", "1I", "5S", "8B", "2Z", "-/"]}


def code_distance(a: str, b: str) -> float:
    """Edit distance where OCR-confusable substitutions cost 0.4."""
    a, b = norm_code(a), norm_code(b)
    n, m = len(a), len(b)
    d = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub = 0.0 if a[i - 1] == b[j - 1] else (0.4 if frozenset((a[i - 1], b[j - 1])) in CONFUSABLE else 1.0)
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + sub)
    return d[n][m]


def norm_code(s: str) -> str:
    s = (s or "").upper().translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789"))
    return re.sub(r"[\s./_]", "-", s).strip("-")


class Registry:
    """Patient profiles keyed by a random internal id; never derived from personal data."""

    def __init__(self):
        self.patients: dict[str, dict] = {}

    def create(self, code: str, facts: dict) -> str:
        pid = uuid.uuid4().hex
        self.patients[pid] = dict(code=norm_code(code), facts=dict(facts), visits=[])
        return pid

    def candidates(self, code: str, facts: dict, max_dist=2.0, k=2):
        out = []
        for pid, p in self.patients.items():
            dist = code_distance(code, p["code"])
            if dist > max_dist:
                continue
            agree = sum(1 for f, v in facts.items() if v and p["facts"].get(f) == v)
            clash = sum(1 for f, v in facts.items() if v and p["facts"].get(f) and p["facts"][f] != v)
            score = -dist + 0.8 * agree - 1.0 * clash
            out.append(dict(pid=pid, code=p["code"], dist=dist, agree=agree, clash=clash, score=score))
        out.sort(key=lambda c: -c["score"])
        return out[:k]

    def propose(self, code: str, facts: dict):
        """Return the question for the midwife. Never auto-creates when a match is plausible."""
        cands = self.candidates(code, facts)
        if cands and cands[0]["dist"] == 0 and cands[0]["clash"] == 0 and \
                (len(cands) == 1 or cands[1]["score"] < cands[0]["score"] - 1):
            buttons = ["Patiente 1", "Aucune, créer", "Je ne sais pas"]
            return dict(kind="CONFIRM_MATCH", candidates=cands[:1], buttons=buttons)
        if cands:
            buttons = [f"Patiente {i + 1}" for i in range(len(cands))] + ["Aucune, créer", "Je ne sais pas"]
            return dict(kind="CHOOSE_MATCH", candidates=cands, buttons=buttons)
        return dict(kind="NO_MATCH", candidates=[], buttons=["Créer la patiente", "Je ne sais pas"])

    def decide(self, proposal: dict, choice: str, code: str, facts: dict):
        """Apply the midwife's button. Returns (pid or None, record state)."""
        if choice.startswith("Patiente"):
            i = int(choice.split()[-1]) - 1
            return proposal["candidates"][i]["pid"], "PATIENTE_LIÉE"
        if choice in ("Aucune, créer", "Créer la patiente"):
            return self.create(code, facts), "PATIENTE_LIÉE"
        return None, "RÉVISION_MANUELLE_REQUISE"      # "Je ne sais pas": nothing linked, nothing created
