"""Counts only: how slab callouts fare in the position tier (no content)."""
import sys
from collections import Counter
from pathlib import Path

from l2c_rebar.config import Config
from l2c_rebar.pipeline import run_project

if __name__ == "__main__":
    proj = Path(sys.argv[1])
    kwargs = {}
    for arg in sys.argv[2:]:
        key, value = arg.split("=")
        kwargs[key] = float(value)
    out = run_project(proj, Path("outputs") / proj.name, Config(ocr="auto", **kwargs))
    c = Counter()
    for r in out.results:
        if r.type_element not in ("dalle", "fondation") or r.plan is None:
            continue
        bare = all(b.diametre is None for b in r.plan.bars)
        kind = "bare count" if bare else "with size"
        c[(r.type_element, kind, r.methode, r.statut)] += 1
        for e in r.ecarts:
            if e.attribut == "quantite":
                diff = e.atelier - e.plan
                c[(r.type_element, kind, r.methode, "quantity diff", "-3+" if diff <= -3 else str(diff) if diff < 0 else "+" + str(min(diff, 3)))] += 1
            else:
                c[(r.type_element, kind, r.methode, "ecart", e.attribut)] += 1
    grid = Counter()
    for e in out.plan_elements + out.shop_elements:
        if e.type_element in ("dalle", "fondation"):
            grid[(e.source, e.type_element, "on grid" if e.grid else "no grid")] += 1
    for k, v in sorted(c.items(), key=str):
        print("  ", k, v)
    print("   grid coverage:", dict(grid))
