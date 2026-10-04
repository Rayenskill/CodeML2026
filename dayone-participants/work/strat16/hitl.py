"""Strategy 16: human-in-the-loop learning, simulated.

Every reviewed field yields (signals, was the extracted value right?). We replay a stream of pages:
  * the confidence threshold tau for auto-accept is re-selected after each batch of reviewed pages so that
    auto-accepted fields stay >= TARGET accurate on everything reviewed so far (online recalibration);
  * the logistic calibrator is refit on accumulated feedback (cold start: uncalibrated CTC confidence).
Reported: questions per page and auto-accept accuracy over time (learning curve).

python hitl.py --rows ~/dayone_local/cal_rows.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

W = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(W / "strat6")]
from calibrate import featurize  # noqa: E402

TARGET = 0.99


def choose_tau(p, y, target=TARGET):
    order = np.argsort(-p)
    ps, ys = p[order], y[order]
    acc = np.cumsum(ys) / np.arange(1, len(ys) + 1)
    ok = np.where(acc >= target)[0]
    if len(ok) == 0:
        return 1.01
    return float(ps[ok.max()])


def simulate(rows, fields_per_page=60, batch_pages=10, seed=0):
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(seed)
    rows = [rows[i] for i in rng.permutation(len(rows))]
    X = np.array([featurize(r["feats"], r["kind"]) for r in rows])
    y = np.array([r["ok"] for r in rows], int)
    n_pages = len(rows) // fields_per_page
    curve = []
    tau, lr = 0.999, None
    seen = 0
    for b in range(0, n_pages, batch_pages):
        lo, hi = b * fields_per_page, min((b + batch_pages) * fields_per_page, len(rows))
        Xb, yb = X[lo:hi], y[lo:hi]
        p = lr.predict_proba(Xb)[:, 1] if lr is not None else np.exp(Xb[:, 0])
        auto = p >= tau
        curve.append(dict(pages=hi // fields_per_page, questions_per_page=float((~auto).sum() / ((hi - lo) / fields_per_page)),
                          auto_acc=float(yb[auto].mean()) if auto.any() else float("nan"), tau=tau))
        seen = hi
        if len(set(y[:seen])) == 2:
            lr = LogisticRegression(max_iter=1000).fit(X[:seen], y[:seen])
            tau = choose_tau(lr.predict_proba(X[:seen])[:, 1], y[:seen])
    return curve


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--rows", required=True)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.rows, encoding="utf-8")]
    for c in simulate(rows):
        print(c)
