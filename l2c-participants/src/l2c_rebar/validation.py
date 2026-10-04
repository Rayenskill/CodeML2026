"""Engineer's verdicts on uncertain findings (bonus: confidence score + validation)."""

from __future__ import annotations

import json
from pathlib import Path

from .models import Result

CONFIRMED = "confirme"
REJECTED = "rejete"


def load_validations(path: Path) -> dict[str, dict]:
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def save_validations(path: Path, validations: dict[str, dict]) -> None:
    Path(path).write_text(json.dumps(validations, ensure_ascii=False, indent=1), encoding="utf-8")


def result_key(r: Result) -> str:
    """Stable across runs, unlike the NC-0001 numbering."""
    return f"{r.statut}|{r.plan.id if r.plan else ''}|{r.atelier.id if r.atelier else ''}"


def apply_validations(results: list[Result], validations: dict[str, dict]) -> list[Result]:
    """Drop the findings the engineer rejected; mark the confirmed ones as certain."""
    kept = []
    for r in results:
        verdict = validations.get(result_key(r), {}).get("verdict")
        if verdict == REJECTED:
            continue
        if verdict == CONFIRMED:
            r.confiance = 1.0
            r.note = (r.note + " " if r.note else "") + "Validé par l'ingénieur."
        kept.append(r)
    return kept
