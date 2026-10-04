"""Strategy 6 T3/T4: per-status confusion matrix with all statuses represented.

Synthetic pages (fresh seeds, all fonts) already contain CONNU values, blanks, written dashes and INCONNU
answers ("?", "inconnu", "NSP"...). We add ILLISIBLE cases by scribbling over ~6% of the written values
(dense random pen strokes over the ink, like a crossed-out entry) and NON_APPLICABLE cases come from the form
logic. Every page goes through degradation (sev 0-2) and the full pipeline.
"never hide doubt" (T4): share of scribbled fields predicted CONNU must be ~0.

python status_bench.py --pages 40
"""
import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

W = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(W / s) for s in ("shared", "strat2", "strat4", "strat5", "strat6", "strat7", "strat11", "strat18")]
STATUSES = ["CONNU", "À_RÉVISER", "ILLISIBLE", "INCONNU", "NON_FOURNI", "NON_APPLICABLE"]


def scribble(page_img, bbox, rng, ink):
    im = Image.fromarray(page_img); d = ImageDraw.Draw(im)
    x0, y0, x1, y1 = bbox
    for _ in range(int(rng.integers(6, 12))):
        pts = [(rng.uniform(x0 - 4, x1 + 4), rng.uniform(y0 - 3, y1 + 3)) for _ in range(5)]
        d.line(pts, fill=ink, width=int(rng.integers(2, 4)))
    return np.asarray(im)


def main():
    from common import MODELS
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=int, default=40)
    ap.add_argument("--seed", type=int, default=91000)
    ap.add_argument("--cal", default=str(MODELS / "calibrator.json"))
    a = ap.parse_args()
    from calibrate import Calibrator
    from degrade import degrade_page
    from extract_zonal import Extractor
    from omr import OMR
    from recognizer import Recognizer
    from synth_pages import synth_page
    from validator import apply_form_logic
    ext = Extractor(Recognizer(str(MODELS / "crnn_final.pt")), OMR(str(MODELS / "omr_v2.pt")), Calibrator.load(a.cal))
    rng = np.random.default_rng(a.seed)
    conf = collections.Counter()
    for i in range(a.pages):
        t = int(rng.choice([2, 3, 4, 5, 6, 7], p=[0.15, 0.35, 0.1, 0.15, 0.1, 0.15]))
        page, fields, meta = synth_page(t, rng, holdout_specimen=False)
        gt = {}
        for f in fields:
            if f["type"] != "text":
                continue
            st = f["status"]
            if f["value"] and st == "CONNU" and f.get("bbox") and rng.random() < 0.06:
                page = scribble(page, f["bbox"], rng, meta["ink"]); st = "ILLISIBLE"
            gt[f["key"]] = st
        g = dict(page_type=t, fields=[dict(key=k, value=None if s != "CONNU" else "x", status=s, type="text") for k, s in gt.items()]
                 + [dict(key=f["key"], type="checkbox", value=f["value"], status="CONNU") for f in fields if f["type"] == "checkbox"])
        apply_form_logic(g)
        gt = {f["key"]: f["status"] for f in g["fields"] if f["type"] == "text"}
        photo, _, _ = degrade_page(page, int(rng.integers(0, 3)), rng)
        pred = ext.extract(photo, t)
        for f in pred["fields"]:
            if f["type"] == "text" and f["key"] in gt:
                conf[(gt[f["key"]], f["status"])] += 1
        print(i + 1, flush=True) if (i + 1) % 10 == 0 else None
    print("\nrows = ground truth, columns = predicted")
    print("| GT \\ pred | " + " | ".join(STATUSES) + " | n |")
    print("|---" * (len(STATUSES) + 2) + "|")
    for g in STATUSES:
        n = sum(conf[(g, p)] for p in STATUSES)
        if n:
            print(f"| {g} | " + " | ".join(f"{conf[(g, p)] / n:.2f}" for p in STATUSES) + f" | {n} |")
    ill = sum(conf[("ILLISIBLE", p)] for p in STATUSES)
    print(f"\nnever-hide-doubt: scribbled fields predicted CONNU = {conf[('ILLISIBLE', 'CONNU')]}/{ill}")
    (Path.home() / "dayone_local" / "status_bench.json").write_text(json.dumps({f"{k[0]}->{k[1]}": v for k, v in conf.items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
