"""Strategy 5 T3/T5: extraction accuracy per language on synthetic pages (the specimen has no Arabic).

Same page types, same seeds for every language (only the language of the written values changes); handwriting fonts
held out or not; photos at a fixed severity; full pipeline with the shipped models, except that the page type is
given (this measures reading, not page typing).

python bench_languages.py --pages 24 --sev 0 2 [--holdout 1]
"""
import argparse
import collections
import sys
from pathlib import Path

import numpy as np

W = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(W / s) for s in ("shared", "strat2", "strat4", "strat5", "strat6", "strat7", "strat11", "strat18")]
from common import MODELS  # noqa: E402
from vocab import canonical  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=int, default=24)
    ap.add_argument("--sev", type=int, nargs="+", default=[0, 2])
    ap.add_argument("--holdout", type=int, default=0, help="1 = only fonts the recogniser may not have seen")
    ap.add_argument("--seed", type=int, default=77000)
    a = ap.parse_args()
    from calibrate import Calibrator
    from degrade import degrade_page
    from extract_zonal import Extractor
    from omr import OMR
    from recognizer import Recognizer
    from synth_pages import synth_page
    ext = Extractor(Recognizer(str(MODELS / "crnn_final.pt")), OMR(str(MODELS / "omr_v2.pt")),
                    Calibrator.load(str(MODELS / "calibrator.json")))
    res = collections.defaultdict(lambda: [0, 0])
    for sev in a.sev:
        for lang in ("fr", "en", "ar"):
            for i in range(a.pages):
                rng = np.random.default_rng(a.seed + i)          # same page type / layout draw for every language
                t = int(rng.integers(1, 9))
                page, fields, _ = synth_page(t, rng, holdout_specimen=bool(a.holdout), lang=lang, printed=False)
                photo = degrade_page(page, sev, np.random.default_rng(a.seed + 10_000 + i))[0] if sev else page
                pred = {f["key"]: f for f in ext.extract(photo, t)["fields"]}
                for f in fields:
                    if f["type"] == "text" and f["value"] and f["key"] in pred:
                        ok = canonical(pred[f["key"]].get("value")) == canonical(f["value"])
                        res[(sev, lang)][0] += ok
                        res[(sev, lang)][1] += 1
            ok, n = res[(sev, lang)]
            print(f"sev {sev} {lang}: filled accuracy {ok / max(n, 1):.3f} ({n} values)", flush=True)
    print("| sev | FR | EN | AR |\n|---|---|---|---|")
    for sev in a.sev:
        print(f"| {sev} | " + " | ".join(f"{res[(sev, l)][0] / max(res[(sev, l)][1], 1):.3f}" for l in ("fr", "en", "ar")) + " |")


if __name__ == "__main__":
    main()
