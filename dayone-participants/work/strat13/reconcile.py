"""Strategy 13: longitudinal reconciliation of a re-photographed page against the patient's validated record.

Field classes: stable (dates of pregnancy, blood group, history...), append-only (past visit columns),
new (empty before). Decision per field:
  equal                         -> keep, confidence up
  stable/append, edit dist <= 2 -> keep the validated value (misread of the same ink), log
  stable/append, different      -> À_RÉVISER with both values ([Garder l'ancien] [Prendre le nouveau])
  new (empty before)            -> normal extraction status
  filled before, blank now      -> keep validated (ink does not disappear), flag if confident blank
"""
from __future__ import annotations

from rapidfuzz.distance import Levenshtein


def field_class(key: str) -> str:
    if key.startswith("visites."):
        return "append"
    if key.startswith("cb."):
        return "stable"
    return "stable"


def reconcile(validated: dict, new: dict, max_edit=2):
    """validated / new: page dicts with fields [{key, value, status, confidence}]. Returns (merged page, stats)."""
    old = {f["key"]: f for f in validated["fields"]}
    out, stats = [], dict(kept=0, repaired=0, conflicts=0, new=0)
    for f in new["fields"]:
        o = old.get(f["key"])
        g = dict(f)
        ov, nv = (o or {}).get("value"), f.get("value")
        if o is None or ov in (None, False) and f.get("type") != "checkbox":
            if nv:
                stats["new"] += 1
        elif ov == nv:
            g["confidence"] = max(f.get("confidence", 0), 0.99); g["status"] = "CONNU" if nv else g["status"]
            stats["kept"] += 1
        elif f.get("type") == "checkbox":
            g.update(status="À_RÉVISER", previous=ov); stats["conflicts"] += 1
        elif nv is None or Levenshtein.distance(str(ov), str(nv)) <= max_edit:
            g.update(value=ov, status="CONNU", confidence=0.99, provenance="validated_record"); stats["repaired"] += 1
        else:
            g.update(status="À_RÉVISER", previous=ov, suggestions=[ov]); stats["conflicts"] += 1
        out.append(g)
    return dict(new, fields=out), stats
