"""Counts-only: what disagrees in pairs made by name, per project (no content)."""
import sys
from collections import Counter
from pathlib import Path
from l2c_rebar.pipeline import run_project
from l2c_rebar.config import Config
if __name__ == "__main__":
    proj = Path(sys.argv[1])
    out = run_project(proj, Path("outputs") / proj.name, Config(min_label_agreement=0.0))
    c = Counter(); bars = Counter(); roles = Counter(); qd = Counter(); nb = Counter()
    for r in out.results:
        if r.methode != "repere" or r.type_element != "colonne" or not r.atelier: continue
        c[r.statut] += 1
        nb[("plan bars", len(r.plan.bars))] += 1; nb[("shop bars", min(len(r.atelier.bars), 8))] += 1
        for e in r.ecarts:
            role = e.plan_bar.role if e.plan_bar is not None else None
            c[("ecart", e.attribut, e.gravite, role)] += 1
            if e.attribut == "quantite" and isinstance(e.atelier, int) and isinstance(e.plan, int):
                ratio = e.atelier / e.plan
                qd["atelier = k x plan" if ratio >= 2 and abs(ratio - round(ratio)) < 1e-9 else "atelier = plan / k" if ratio <= 0.5 and abs(1 / ratio - round(1 / ratio)) < 1e-9 else "atelier > plan" if ratio > 1 else "atelier < plan"] += 1
        for b in r.atelier.bars:
            bars[("shop bar has", "qty" if b.quantite else "-", "esp" if b.espacement_mm else "-", "len" if b.longueur_mm else "-", "mark" if b.repere else "-")] += 1
    for k, v in sorted(c.items(), key=str): print("  ", k, v)
    print("   quantity direction:", dict(qd))
    print("   bars per element:", sorted(nb.items()))
    print("   shop bar fields:", bars.most_common(6))
    print("   level shift:", out.coverage.level_shift.get("colonne"), "| totals:", out.coverage.totaux)
