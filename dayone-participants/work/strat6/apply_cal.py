"""Apply a fitted calibrator to saved specimen predictions and report uncertainty metrics (strategy 6, T1-T3, T5).

python apply_cal.py --tag final_sev0 --cal cal_final.joblib [--tau_high 0.9 --tau_low 0.35]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

W = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(W / s) for s in ("shared", "strat5", "strat6", "strat7")]
from calibrate import Calibrator  # noqa: E402
from common import GT_DIR, LOCAL  # noqa: E402
from eval import ece  # noqa: E402
from fieldlogic import kind, status_for  # noqa: E402
from vocab import canonical  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--tag"); ap.add_argument("--cal")
ap.add_argument("--tau_high", type=float, default=0.95); ap.add_argument("--tau_low", type=float, default=0.35)
a = ap.parse_args()
cal = Calibrator.load(a.cal)
d = json.loads((LOCAL / "preds" / f"{a.tag}.json").read_text(encoding="utf-8"))
conf, ok, st_ok, n_pages, asks = [], [], 0, 0, 0
n_st = 0
for pid, p in d["preds"].items():
    g = json.loads((GT_DIR / f"page_{int(pid):02d}.json").read_text(encoding="utf-8"))
    gt = {f["key"]: f for f in g["fields"]}
    n_pages += 1
    for f in p["fields"]:
        if f["type"] != "text" or "feats" not in f:
            continue
        gf = gt.get(f["key"])
        if gf is None or gf.get("identifier"):
            continue
        c = cal.predict(f["feats"], kind(p["page_type"], f["key"])["kind"])
        good = canonical(f["value"]) == canonical(gf["value"])
        conf.append(c); ok.append(good)
        s = status_for(f["value"], c, a.tau_high, a.tau_low)
        if f.get("repairs") or f.get("flags"):
            s = "À_RÉVISER" if f.get("flags") else s
        asks += s in ("À_RÉVISER", "ILLISIBLE")
        gs = gf["status"]
        n_st += 1
        # a correct value may be shown as À_RÉVISER (doubt is allowed); a wrong value must never be CONNU
        st_ok += (s == gs) or (gs == "CONNU" and s == "À_RÉVISER") or (gs == "CONNU" and s == "ILLISIBLE" and not good)
conf, ok = np.array(conf), np.array(ok, float)
auto = conf >= a.tau_high
wrong_but_connu = int(((conf >= a.tau_high) & (ok == 0)).sum())
print(f"{a.tag}: fields {len(ok)} acc {ok.mean():.4f} | ECE {ece(conf, ok):.4f} | status_acc {st_ok / n_st:.4f} | "
      f"auto-accept {auto.mean():.3f} at accuracy {ok[auto].mean():.4f} | wrong-but-CONNU {wrong_but_connu} | "
      f"questions/page {asks / n_pages:.1f}")
