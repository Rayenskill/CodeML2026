"""Masked, aggregated notation shapes on shop-drawing pages read by OCR.
Digits -> 9 (length kept up to 3), words -> A, single letters -> a, bar sizes -> #M.
Only shapes seen >= MIN times are printed: no value, name or one-off string appears."""
import re, sys
from collections import Counter, defaultdict
from pathlib import Path
from l2c_rebar.config import Config
from l2c_rebar.pipeline import discover, shop_file_hints
from l2c_rebar.extract.document import read_document
from l2c_rebar.parsing.rebar import normalize_ocr
MIN = 8
SIZE = re.compile(r"(?<![0-9A-Za-z.,/])(?:10|15|20|25|30|35|45|55)\s?M(?![A-Za-z])")
def mask(t):
    t = SIZE.sub("\x00", t)
    t = re.sub(r"[A-Za-zÀ-ÿ]+", lambda m: "A" if len(m.group()) > 1 else "a", t)
    t = re.sub(r"\d+", lambda m: "9" * min(len(m.group()), 3), t)
    return re.sub(r"\s+", " ", t).strip().replace("\x00", "#M")
if __name__ == "__main__":
    proj = Path(sys.argv[1]); want = set(sys.argv[2].split(",")); cfg = Config(); cache = Path("outputs") / proj.name / ".ocr-cache"
    files = discover(proj); sized = defaultdict(Counter); short = defaultdict(Counter); n = Counter()
    for p in files.shops:
        th, lv = shop_file_hints(p, files.root)
        if th not in want: continue
        for pg in read_document(p, "atelier", cfg, cache, th, lv)[0]:
            n[(th, pg.origin)] += 1
            for line in pg.lines:
                t = normalize_ocr(line.text.strip()) if line.origin == "ocr" else line.text.strip()
                if SIZE.search(t) and len(t) <= 60: sized[th][mask(t)] += 1
                elif len(t) <= 22: short[th][mask(t)] += 1
    print("pages:", dict(n))
    for th in sorted(sized):
        c = sized[th]; print(f"[{th}] callout lines={sum(c.values())} distinct={len(c)}")
        for s, k in c.most_common(16):
            if k >= MIN: print(f"    {k:5d}  {s}")
        c = short[th]; print(f"[{th}] short lines={sum(c.values())}: " + " | ".join(f"{s} x{k}" for s, k in c.most_common(30) if k >= MIN))
