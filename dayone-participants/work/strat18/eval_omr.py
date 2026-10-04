"""Checkbox accuracy of OMR models on the specimen pages (registered with the full pipeline)."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

W = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(W / s) for s in ("shared", "strat2", "strat4", "strat18")]
from common import GT_DIR, REG  # noqa: E402
from crops import checkbox_crop  # noqa: E402
from degrade import degrade_page  # noqa: E402
from omr import OMR  # noqa: E402
from register import register  # noqa: E402

models = {Path(p).stem: OMR(p) for p in sys.argv[2:]}
sev = int(sys.argv[1])
rng = np.random.default_rng(100 + sev)
ok = {k: 0 for k in models}; n = 0; per_pat = {k: {} for k in models}
for pg in range(1, 81):
    g = json.loads((GT_DIR / f"page_{pg:02d}.json").read_text(encoding="utf-8"))
    img = cv2.cvtColor(cv2.imread(str(REG / g["png"][0])), cv2.COLOR_BGR2RGB)
    photo = degrade_page(img, sev, rng)[0] if sev else img
    r = register(photo, g["page_type"])
    cbs = [f for f in g["fields"] if f["type"] == "checkbox"]
    crops = [checkbox_crop(r["warped"], g["page_type"], f["key"])[0] for f in cbs]
    for k, m in models.items():
        p = m.predict_many(crops)
        c = int(((p >= 0.5) == np.array([f["value"] for f in cbs])).sum())
        ok[k] += c
        a = per_pat[k].setdefault(g["patient"], [0, 0]); a[0] += c; a[1] += len(cbs)
    n += len(cbs)
for k in models:
    print(f"sev {sev} {k}: {ok[k] / n:.4f} (n={n})", {p: round(a / b, 3) for p, (a, b) in per_pat[k].items()})
