"""Masked, aggregated shapes of short non-callout lines (label / header formats).
Digits -> 9, letters -> A (runs collapsed); only shapes seen >= MIN_COUNT times. No values, no names."""
import sys, re
from collections import Counter, defaultdict
from pathlib import Path
from l2c_rebar.config import Config
from l2c_rebar.pipeline import discover, shop_file_hints
from l2c_rebar.extract.document import read_document
MIN_COUNT = 8
SIZE = re.compile(r"(?<![0-9A-Za-z.,/])(?:10|15|20|25|30|35|45|55)\s?M(?![A-Za-z])")
def mask(t):
    t = re.sub(r"[A-Za-zÀ-ÿ]+", lambda m: "A" if len(m.group()) > 1 else "a", t)   # a = single letter, A = word
    t = re.sub(r"\d+", lambda m: "9" * min(len(m.group()), 4), t)                   # keep number of digits (max 4)
    return re.sub(r"\s+", " ", t).strip()
proj = Path(sys.argv[1]); types = set(sys.argv[2].split(","))
files = discover(proj); cache = Path("outputs") / proj.name / ".ocr-cache"
docs = [(p, "plan", None, ()) for p in files.plans] + [(p, "atelier", *shop_file_hints(p, files.root)) for p in files.shops]
tab = defaultdict(Counter); paren = defaultdict(Counter)
P = re.compile(r"(\d{1,3})\s?\(\s?(\d{1,5})\s?\)\s?-\s?(?:10|15|20|25|30|35)M")
for path, source, th, lv in docs:
    pages, _ = read_document(path, source, Config(ocr="auto"), cache if cache.exists() else None, th, lv)
    for pg in pages:
        if pg.type_element not in types: continue
        for line in pg.lines:
            t = line.text.strip()
            for m in P.finditer(t):
                q, k = int(m.group(1)), int(m.group(2))
                paren[(source, pg.type_element)][f"paren_digits={len(m.group(2))}"] += 1
                paren[(source, pg.type_element)]["paren<=qty" if k <= q else "paren>qty"] += 1
            if not SIZE.search(t) and len(t) <= 26:
                tab[(source, pg.type_element, pg.origin)][mask(t)] += 1
for key in sorted(tab):
    c = tab[key]; print(key, "short lines:", sum(c.values()), "distinct:", len(c))
    print("    " + " | ".join(f"{s} x{n}" for s, n in c.most_common(28) if n >= MIN_COUNT))
for key in sorted(paren): print("paren stats", key, dict(paren[key]))
