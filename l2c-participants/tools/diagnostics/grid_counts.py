"""Counts-only: how many column tags get a grid name, and how many names are shared with the shop drawings."""
import sys
from collections import Counter
from pathlib import Path
from l2c_rebar.config import Config
from l2c_rebar.pipeline import discover, shop_file_hints
from l2c_rebar.extract.document import read_document, extract_pages
if __name__ == "__main__":
    proj = Path(sys.argv[1]); cfg = Config(ocr="auto"); cache = Path("outputs") / proj.name / ".ocr-cache"
    files = discover(proj)
    plan_pages, _ = read_document(files.plans[0], "plan", cfg, None)
    shop_pages = []
    for p in files.shops:
        th, lv = shop_file_hints(p, files.root)
        if th == "colonne": shop_pages += read_document(p, "atelier", cfg, cache, th, lv)[0]
    for pg in plan_pages:
        if pg.type_element == "colonne":
            print(f"plan p{pg.page_index+1}: vertical lines={len(pg.axes.vertical)} horizontal lines={len(pg.axes.horizontal)} letters vertical={pg.axes.letters_are_vertical} marks={len(pg.marks)} segments={len(pg.segments)} levels={len(pg.levels)}")
    plan = [e for e in extract_pages(plan_pages, cfg, {}, None) if e.type_element == "colonne"]
    shop = [e for e in extract_pages(shop_pages, cfg, {}, None) if e.type_element == "colonne"]
    pl = {e.label for e in plan if e.label}; sl = {e.label for e in shop if e.label}
    print(f"plan column elements={len(plan)} named={sum(1 for e in plan if e.label)} distinct names={len(pl)} with '-'={sum(1 for l in pl if '-' in l)}")
    print(f"shop column elements={len(shop)} named={sum(1 for e in shop if e.label)} distinct names={len(sl)} with '-'={sum(1 for l in sl if '-' in l)} | shop pages={len(shop_pages)} ocr pages={sum(p.origin=='ocr' for p in shop_pages)}")
    print(f"names in common={len(pl & sl)} | plan only={len(pl - sl)} | shop only={len(sl - pl)}")
    print("bars per named plan element:", Counter(len(e.bars) for e in plan if e.label).most_common(4), "| per named shop element:", Counter(min(len(e.bars), 12) for e in shop if e.label).most_common(6))
    print("shop levels per named element:", Counter(len(e.levels) for e in shop if e.label).most_common(4), "| shop elements with schedule level:", sum(1 for e in shop for b in e.bars[:1] if b.level))
