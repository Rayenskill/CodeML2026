"""Notation shapes of rebar callout lines, fully masked and aggregated.

Every digit run becomes 9, every letter run becomes A, the bar size becomes #M.
Only shapes seen at least MIN_COUNT times are printed, so no value, name or
one-off string from a drawing can appear: only the recurring notation pattern.
"""
import sys, re
from collections import Counter, defaultdict
from pathlib import Path
from l2c_rebar.config import Config
from l2c_rebar.pipeline import discover, shop_file_hints
from l2c_rebar.extract.document import read_document

MIN_COUNT = 5
SIZE = re.compile(r"(?<![0-9A-Za-z.,/])(?:10|15|20|25|30|35|45|55)\s?M(?![A-Za-z])")
def mask(t):
    t = SIZE.sub("\x00", t)
    t = re.sub(r"[A-Za-zÀ-ÿ]+", "A", t)
    t = re.sub(r"\d+", "9", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t.replace("\x00", "#M")

proj = Path(sys.argv[1]); cfg = Config(ocr="off"); cache = Path("outputs") / proj.name / ".ocr-cache"
files = discover(proj)
docs = [(p, "plan", None, ()) for p in files.plans] + [(p, "atelier", *shop_file_hints(p, files.root)) for p in files.shops]
tab = defaultdict(Counter)
for path, source, th, lv in docs:
    pages, _ = read_document(path, source, Config(ocr="auto"), cache if cache.exists() else None, th, lv)
    for pg in pages:
        for line in pg.lines:
            if SIZE.search(line.text) and len(line.text) <= 60:
                tab[(source, pg.type_element, pg.origin)][mask(line.text)] += 1
for key in sorted(tab):
    c = tab[key]; total = sum(c.values()); shown = [(s, n) for s, n in c.most_common(14) if n >= MIN_COUNT]
    print(key, "lines:", total, "distinct shapes:", len(c))
    for s, n in shown: print(f"    {n:5d}  {s}")
