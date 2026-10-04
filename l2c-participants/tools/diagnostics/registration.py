"""Counts only: is the grid registration between plan and shop slab sheets right?

For each plan callout carrying a size and a quantity, distance (in grid bays) to the nearest shop
callout of the same storey with the identical size and quantity.  A sound registration piles up
near zero.  Also prints, per shop page, how many grid lines were read (no labels printed)."""
import sys
from collections import Counter
from pathlib import Path

from l2c_rebar.config import Config
from l2c_rebar.extract.document import read_document
from l2c_rebar.pipeline import discover, run_project, shop_file_hints


def bucket(d: float) -> str:
    for limit in (0.25, 0.5, 1.0, 2.0, 4.0):
        if d <= limit:
            return f"<={limit}"
    return ">4"


if __name__ == "__main__":
    proj = Path(sys.argv[1])
    kind = sys.argv[2] if len(sys.argv) > 2 else "dalle"
    cfg = Config(ocr="auto")
    out = run_project(proj, Path("outputs") / proj.name, cfg)
    plan = [e for e in out.plan_elements if e.type_element == kind and e.grid]
    shop = [e for e in out.shop_elements if e.type_element == kind and e.grid]
    hist, dx, dy = Counter(), Counter(), Counter()
    for p in plan:
        for b in p.bars:
            if not (b.diametre and b.quantite and b.quantite >= 3):
                continue
            best = None
            for s in shop:
                if p.levels and s.levels and not set(p.levels) & set(s.levels):
                    continue
                if any(sb.diametre == b.diametre and sb.quantite == b.quantite for sb in s.bars):
                    a, c = (p.grid[0] - s.grid[0]) / p.grid[2], (p.grid[1] - s.grid[1]) / p.grid[3]
                    d = (a * a + c * c) ** 0.5
                    if best is None or d < best[0]:
                        best = (d, a, c)
            if best:
                hist[bucket(best[0])] += 1
                if best[0] <= 4:
                    dx[round(best[1] * 2) / 2] += 1
                    dy[round(best[2] * 2) / 2] += 1
            else:
                hist["none"] += 1
    print("nearest identical shop callout, in bays:", dict(sorted(hist.items())))
    print("letter-axis offset (bays, top 6):", dx.most_common(6))
    print("number-axis offset (bays, top 6):", dy.most_common(6))
    cache = Path("outputs") / proj.name / ".ocr-cache"
    files = discover(proj)
    pages = []
    for f in files.shops:
        hint, levels = shop_file_hints(f, files.root)
        if hint == kind:
            pages += read_document(f, "atelier", cfg, cache, hint, levels)[0]
    print("shop pages: (vertical lines, horizontal lines, letters vertical):",
          Counter((len(pg.axes.vertical), len(pg.axes.horizontal), pg.axes.letters_are_vertical) for pg in pages).most_common(8))
    plan_pages, _ = read_document(files.plans[0], "plan", cfg, None)
    print("plan pages: ", Counter((len(pg.axes.vertical), len(pg.axes.horizontal), pg.axes.letters_are_vertical)
                                  for pg in plan_pages if pg.type_element == kind).most_common(8))
