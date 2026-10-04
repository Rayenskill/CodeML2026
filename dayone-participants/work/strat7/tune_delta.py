"""Tune the vocabulary-snapping margin `delta` offline from cached CTC log-probs (strategy 7/11).

python tune_delta.py --tag heldout_v1ep2_sev0 --model crnn.pt --patients 1-5
Leave-one-patient-out vocabularies; text fields only (checkboxes unaffected).
"""
import argparse
import json
import sys
from pathlib import Path

import torch

W = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(W / s) for s in ("shared", "strat5", "strat7", "strat11")]
import fieldlogic  # noqa: E402
from common import GT_DIR, LOCAL  # noqa: E402
from recognizer import Recognizer  # noqa: E402
from vocab import canonical  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--tag"); ap.add_argument("--model"); ap.add_argument("--patients", default="1-10")
ap.add_argument("--deltas", default="0,1,2.5,4,6,8,10,14")
a = ap.parse_args()
lo, hi = map(int, a.patients.split("-"))
rec = Recognizer(a.model)
preds = json.loads((LOCAL / "preds" / f"{a.tag}.json").read_text(encoding="utf-8"))["preds"]
lps = torch.load(LOCAL / "preds" / f"{a.tag}_lp.pt", weights_only=False)
items = []
for pid, p in preds.items():
    g = json.loads((GT_DIR / f"page_{int(pid):02d}.json").read_text(encoding="utf-8"))
    if not lo <= g["patient"] <= hi:
        continue
    gt = {f["key"]: f for f in g["fields"]}
    lp = lps[pid] if pid in lps else lps[int(pid)]
    for f in p["fields"]:
        if f["type"] == "text" and f["key"] in lp:
            items.append((g["patient"], g["page_type"], f["key"], f.get("raw") or "", lp[f["key"]], gt[f["key"]]["value"]))
print("fields", len(items))
for d in [float(x) for x in a.deltas.split(",")]:
    ok = ok_f = n_f = 0
    for pat, t, key, raw, lp, gv in items:
        fieldlogic.EXCLUDE_PATIENT = pat
        v, _ = fieldlogic.decide(rec, lp, raw, t, key, d)
        v = None if v.strip() in ("", "–", "-") else v.strip()
        c = canonical(v) == canonical(gv)
        ok += c
        if gv:
            n_f += 1; ok_f += c
    print(f"delta {d:5.1f}: text_acc {ok / len(items):.4f} filled_acc {ok_f / n_f:.4f}")
