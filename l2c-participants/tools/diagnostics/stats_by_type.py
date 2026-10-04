"""Counts-only run with per-type counters; never prints document text."""
import sys
from collections import Counter
from pathlib import Path
from l2c_rebar.pipeline import run_project
from l2c_rebar.config import Config
if __name__ == "__main__":
    proj = Path(sys.argv[1]); ocr = sys.argv[2] if len(sys.argv) > 2 else "off"
    out = run_project(proj, Path("outputs") / proj.name, Config(ocr=ocr))
    keys = ["plan_elements", "plan_reperes", "atelier_elements", "atelier_reperes", "reperes_communs", "conforme", "non_conforme", "manquant_atelier", "ajoute_atelier", "par_repere"]
    print(f"{'type':14s} " + " ".join(k[:11].rjust(11) for k in keys))
    for t, c in out.stats["par_type"].items():
        print(f"{t:14s} " + " ".join(str(c.get(k, 0)).rjust(11) for k in keys))
    cov = out.coverage
    print("level align:", cov.level_shift, "| totals:", cov.totaux, "| lecture:", cov.lecture, "| distrusted:", cov.reperes_non_fiables, "| sans atelier:", cov.types_sans_atelier, "| non couverts:", len(cov.feuillets_sans_atelier), "| atelier non apparies:", cov.atelier_non_apparies)
    c = Counter()
    for r in out.results:
        if r.methode == "repere":
            c[(r.type_element, r.statut, "level match" if (r.plan and r.atelier and set(r.plan.levels) & set(r.atelier.levels)) else "no level match")] += 1
            for e in r.ecarts: c[(r.type_element, "ecart", e.attribut, e.gravite)] += 1
    for k, v in sorted(c.items()): print("  ", k, v)
    lv = Counter(); 
    for e in out.plan_elements:
        if e.type_element == "colonne" and e.label: lv[("plan", len(e.levels))] += 1
    for e in out.shop_elements:
        if e.type_element == "colonne" and e.label: lv[("atelier", len(e.levels))] += 1
    print("levels per named column element:", dict(lv))
    pl = {l for e in out.plan_elements if e.type_element == "colonne" for l in e.levels}; sl = {l for e in out.shop_elements if e.type_element == "colonne" for l in e.levels}
    print("distinct plan levels:", len(pl), "distinct shop levels:", len(sl), "in common:", len(pl & sl))
