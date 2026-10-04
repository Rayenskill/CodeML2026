"""Shared paths and helpers for the DayOne extraction work."""
from __future__ import annotations

import os
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # dayone-participants/
DATA = ROOT / "data"
REG = DATA / "Paper Registry"
PDF = REG / "dossiers_specimen_10_patientes.pdf"
WORK = ROOT / "work"
SHARED = WORK / "shared"
GT_DIR = SHARED / "gt"
# Heavy, regenerable artefacts live outside OneDrive (git-ignored by location).
LOCAL = Path(os.environ.get("DAYONE_LOCAL", Path.home() / "dayone_local"))
LOCAL.mkdir(parents=True, exist_ok=True)
# trained models shipped with the repo (dayone-participants/models); override with DAYONE_MODELS
MODELS = Path(os.environ.get("DAYONE_MODELS", ROOT / "models"))

DPI = 200
SCALE = DPI / 72.0           # PDF points -> PNG pixels
PAGE_W, PAGE_H = 1654, 2339

PAGE_TYPES = {
    1: "couverture",
    2: "identification_antecedents",
    3: "grossesse_actuelle",
    4: "accouchement",
    5: "postpartum_precoce_mere",
    6: "postpartum_precoce_nouveau_ne",
    7: "postpartum_tardif_mere",
    8: "postpartum_tardif_nouveau_ne",
}

STATUSES = ["CONNU", "INCONNU", "NON_FOURNI", "ILLISIBLE", "NON_APPLICABLE", "À_RÉVISER"]

# Labels whose value is a direct identifier: never stored, masked before reading.
IDENTIFIER_LABELS = {
    "nom_prenom_de_la_parturiente", "patiente", "cin", "adresse", "telephone", "nom_du_mari",
}


def slug(s: str) -> str:
    s = s.replace("œ", "oe").replace("Œ", "Oe")
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"\.{2,}", " ", s)
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s


def png_for_page(n: int) -> Path:
    """PDF page n (1-based) -> one PNG render (duplicates exist with __suffix)."""
    cands = sorted(REG.glob(f"dossiers_specimen_10_patientes-{n:02d}*.png"))
    return cands[0]
