"""Registration accuracy on clean and degraded specimen pages (strategy 2 §5.2 / strategy 4 bench).

Truth: photo -> template = A(page->ref) o H_true^-1. Error = mean corner displacement (px @200dpi)
of a grid of points over the page.
"""
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

W = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(W / "shared"), str(W / "strat2"), str(W / "strat4")]
from common import GT_DIR, PAGE_H, PAGE_W, REG, SCALE, SHARED  # noqa: E402
from degrade import degrade_page  # noqa: E402
from register import register  # noqa: E402

AFF = json.loads((SHARED / "page_affines.json").read_text())


def page_to_ref(n):
    M = np.array(AFF[str(n)]["M"])
    A = np.eye(3)
    A[:2, :2] = M[:, :2]
    A[:2, 2] = M[:, 2] * SCALE
    return A


def grid_err(H_est, H_true_photo_to_tpl):
    xs, ys = np.meshgrid(np.linspace(100, PAGE_W - 100, 6), np.linspace(100, PAGE_H - 100, 8))
    p = np.stack([xs.ravel(), ys.ravel()], 1).astype(np.float32)[None]
    # take template points back to photo with the truth, then forward with the estimate
    photo = cv2.perspectiveTransform(p, np.linalg.inv(H_true_photo_to_tpl))
    q = cv2.perspectiveTransform(photo, H_est)
    return float(np.linalg.norm(q[0] - p[0], axis=1).mean())


def main(pages=range(1, 81, 2), severities=(0, 1, 2, 3, 4), seed=0):
    rng = np.random.default_rng(seed)
    res = []
    for n in pages:
        g = json.loads((GT_DIR / f"page_{n:02d}.json").read_text(encoding="utf-8"))
        img = cv2.cvtColor(cv2.imread(str(REG / g["png"][0])), cv2.COLOR_BGR2RGB)
        for sev in severities:
            photo, H, _ = degrade_page(img, sev, rng)
            t0 = time.time()
            r = register(photo)
            dt = time.time() - t0
            truth = page_to_ref(n) @ np.linalg.inv(H)
            err = grid_err(r["H"], truth)
            res.append(dict(page=n, sev=sev, type_ok=r["page_type"] == g["page_type"], err=err, cc=r["score"], t=dt))
            print(n, sev, r["page_type"], g["page_type"], round(err, 2), round(r["score"], 3), round(dt, 2), flush=True)
    import pandas as pd
    df = pd.DataFrame(res)
    print(df.groupby("sev").agg(type_acc=("type_ok", "mean"), err_med=("err", "median"), err_p90=("err", lambda x: x.quantile(.9)),
                                err_max=("err", "max"), t=("t", "mean")))
    return df


if __name__ == "__main__":
    main()
