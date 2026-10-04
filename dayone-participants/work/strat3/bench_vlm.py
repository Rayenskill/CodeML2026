"""Compare VLM vs CRNN on the same registered crops of specimen pages (clean or degraded).

python bench_vlm.py --pages 1-16 --sev 0 --crnn crnn.pt [--max 400]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

W = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(W / s) for s in ("shared", "strat2", "strat3", "strat4", "strat5", "strat7", "strat11")]
from common import GT_DIR, REG  # noqa: E402
from crops import crop_zone  # noqa: E402
from degrade import degrade_page  # noqa: E402
from extract_zonal import is_identifier, mask_identifiers  # noqa: E402
from fieldlogic import vocab_for  # noqa: E402
from register import register  # noqa: E402
from vocab import canonical  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", default="1-16"); ap.add_argument("--sev", type=int, default=0)
    ap.add_argument("--crnn"); ap.add_argument("--max", type=int, default=400); ap.add_argument("--filled_only", type=int, default=1)
    ap.add_argument("--out", default=str(Path.home() / "dayone_local" / "vlm_bench.jsonl"))
    a = ap.parse_args()
    a_, _, b_ = a.pages.partition("-")
    pages = range(int(a_), int(b_ or a_) + 1)
    rng = np.random.default_rng(a.sev)
    items = []
    for n in pages:
        g = json.loads((GT_DIR / f"page_{n:02d}.json").read_text(encoding="utf-8"))
        img = cv2.cvtColor(cv2.imread(str(REG / g["png"][0])), cv2.COLOR_BGR2RGB)
        photo = degrade_page(img, a.sev, rng)[0] if a.sev else img
        r = register(photo, g["page_type"])
        warped = mask_identifiers(r["warped"], g["page_type"])
        for f in g["fields"]:
            if f["type"] != "text" or is_identifier(f["key"]):
                continue
            if a.filled_only and not f["value"]:
                continue
            c, _ = crop_zone(warped, g["page_type"], f["key"])
            items.append(dict(page=n, key=f["key"], gt=f["value"], crop=c, t=g["page_type"]))
    items = items[:a.max]
    print("items", len(items), flush=True)
    from recognizer import Recognizer
    rec = Recognizer(a.crnn)
    lps = rec.logprobs([i["crop"] for i in items])
    crnn = [rec.decode(lp)[0].strip() or None for lp in lps]
    from vlm_extract import VLMReader
    t0 = time.time()
    vlm = VLMReader()
    print("vlm loaded", time.time() - t0, flush=True)
    t0 = time.time()
    labels = [i["key"].split(".", 1)[1].replace("_", " ") for i in items]
    vocabs = [list(vocab_for(i["t"], i["key"])) for i in items]
    vr = vlm.read([i["crop"] for i in items], labels, vocabs)
    dt = time.time() - t0
    ok_c = ok_v = 0
    with open(a.out, "w", encoding="utf-8") as fo:
        for it, c, v in zip(items, crnn, vr):
            oc, ov = canonical(c) == canonical(it["gt"]), canonical(v) == canonical(it["gt"])
            ok_c += oc; ok_v += ov
            fo.write(json.dumps(dict(page=it["page"], key=it["key"], gt=it["gt"], crnn=c, vlm=v, ok_c=oc, ok_v=ov),
                                ensure_ascii=False) + "\n")
    n = len(items)
    print(f"sev {a.sev}: CRNN {ok_c / n:.4f}  VLM {ok_v / n:.4f}  (n={n}, VLM {dt / n * 1000:.0f} ms/crop)")


if __name__ == "__main__":
    main()
