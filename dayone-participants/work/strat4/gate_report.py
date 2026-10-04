"""Strategy 4: what the midwife actually gets = accuracy on the photos the on-device quality gate accepts.

The PWA refuses a capture that is blurry (variance of Laplacian at 600 px < 340), dark or glaring and asks for a
retake. This replays the exact degraded photos of a run_eval run (same seeds), applies that gate and scores the
saved predictions on accepted pages only, next to the retake rate.

python gate_report.py --tag release2 --sev 1 2 3 4
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

W = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(W / s) for s in ("shared", "strat2", "strat4", "strat5")]
from common import GT_DIR, LOCAL, REG  # noqa: E402
from degrade import degrade_page  # noqa: E402
from eval import load_gt, score  # noqa: E402
from quality_gate import features  # noqa: E402

GATE_VAR_LAP, GATE_DARK, GATE_GLARE = 340.0, 60.0, 0.08     # same rules as app.quality in work/strat15/app/app.js


def gate_ok(photo) -> bool:
    f = features(photo)
    return f["var_lap"] >= GATE_VAR_LAP and f["mean"] >= GATE_DARK and f["glare"] <= GATE_GLARE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--sev", nargs="+", type=int, default=[1, 2, 3, 4])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    gts = load_gt(GT_DIR)
    rows = []
    for sev in a.sev:
        preds = json.loads((LOCAL / "preds" / f"{a.tag}_sev{sev}.json").read_text(encoding="utf-8"))["preds"]
        preds = {int(k): v for k, v in preds.items()}
        rng = np.random.default_rng(a.seed + sev)
        ok = {}
        for n in sorted(preds):                     # same page order / rng stream as run_specimen
            g = json.loads((GT_DIR / f"page_{n:02d}.json").read_text(encoding="utf-8"))
            img = cv2.cvtColor(cv2.imread(str(REG / g["png"][0])), cv2.COLOR_BGR2RGB)
            photo = degrade_page(img, sev, rng)[0] if sev > 0 else img
            ok[n] = gate_ok(photo)
        acc = {n: p for n, p in preds.items() if ok[n]}
        r_all, _ = score(preds, {k: gts[k] for k in preds})
        r_ok, _ = score(acc, {k: gts[k] for k in acc}) if acc else ({"all": {}}, None)
        rows.append((sev, len(preds), sum(ok.values()), r_all["all"], r_ok["all"]))
    print("| sev | photos accepted | filled (all photos) | filled (accepted) | field (accepted) | status (accepted) |")
    print("|---|---|---|---|---|---|")
    for sev, n, k, ra, ro in rows:
        print(f"| {sev} | {k}/{n} | {ra['filled_acc']:.3f} | {ro.get('filled_acc', float('nan')):.3f} | "
              f"{ro.get('field_acc', float('nan')):.3f} | {ro.get('status_acc', float('nan')):.3f} |")


if __name__ == "__main__":
    main()
