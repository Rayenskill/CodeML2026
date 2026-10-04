"""Strategy 4 (§4.5): on-device quality gate, evaluated against real extraction outcomes.

Replays the exact degraded photos of a run_eval run (same seeds), computes cheap features (the same ones as
the PWA in strategy 15: variance of Laplacian at 600 px, mean brightness, glare fraction, page-quad found /
area), labels a capture "bad" when its filled-field accuracy < 0.6, and reports a depth-2 tree (exportable
as plain rules) with recall of bad captures and false-reject rate.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

W = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(W / s) for s in ("shared", "strat2", "strat4", "strat5")]
from common import GT_DIR, LOCAL, REG  # noqa: E402
from degrade import degrade_page  # noqa: E402
from register import find_page_quad  # noqa: E402
from vocab import canonical  # noqa: E402


def features(rgb):
    h, w = rgb.shape[:2]
    sc = 600 / max(h, w)
    g = cv2.cvtColor(cv2.resize(rgb, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(g.astype(np.float32), cv2.CV_32F).var()
    q = find_page_quad(rgb)
    area = cv2.contourArea(q) / (h * w) if q is not None else 1.0
    return dict(var_lap=float(lap), mean=float(g.mean()), p5=float(np.percentile(g, 5)),
                glare=float((rgb.min(2) > 245).mean()), quad_area=float(area))


def page_acc(pred, g):
    gt = {f["key"]: f for f in g["fields"]}
    ok = n = 0
    for f in pred["fields"]:
        gf = gt.get(f["key"])
        if f["type"] == "text" and gf and gf["value"] and not gf.get("identifier"):
            n += 1; ok += canonical(f["value"]) == canonical(gf["value"])
    return ok / max(n, 1)


def main(tag="final", sevs=(0, 2, 4)):
    rows = []
    for sev in sevs:
        preds = json.loads((LOCAL / "preds" / f"{tag}_sev{sev}.json").read_text(encoding="utf-8"))["preds"]
        rng = np.random.default_rng(0 + sev)           # same stream as run_eval / run_specimen
        for n in range(1, 81):
            g = json.loads((GT_DIR / f"page_{n:02d}.json").read_text(encoding="utf-8"))
            img = cv2.cvtColor(cv2.imread(str(REG / g["png"][0])), cv2.COLOR_BGR2RGB)
            photo = degrade_page(img, sev, rng)[0] if sev > 0 else img
            f = features(photo)
            f.update(sev=sev, page=n, acc=page_acc(preds[str(n)], g))
            rows.append(f)
    import pandas as pd
    from sklearn.tree import DecisionTreeClassifier, export_text
    df = pd.DataFrame(rows)
    df["bad"] = df.acc < 0.6
    X = df[["var_lap", "mean", "p5", "glare", "quad_area"]].values
    tree = DecisionTreeClassifier(max_depth=2, class_weight="balanced", random_state=0).fit(X, df.bad)
    pred = tree.predict(X)
    rec = (pred & df.bad).sum() / max(df.bad.sum(), 1)
    frr = (pred & ~df.bad).sum() / max((~df.bad).sum(), 1)
    print(export_text(tree, feature_names=["var_lap", "mean", "p5", "glare", "quad_area"]))
    print(f"bad captures {int(df.bad.sum())}/{len(df)} | gate recall {rec:.3f} | false rejects {frr:.3f}")
    print(df.groupby("sev")[["var_lap", "mean", "acc"]].median().round(2))
    # leave-one-severity-out check of the same tree depth
    df.to_csv(LOCAL / "quality_gate_rows.csv", index=False)


if __name__ == "__main__":
    main()
