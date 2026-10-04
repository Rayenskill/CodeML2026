"""Strategy 6: calibrated per-field confidence (P(value correct)) and status thresholds.

Signals per field (from extract_zonal): CTC path confidence, min char prob, value length, whether the
value was snapped to the field vocabulary, candidate margin, new-ink fraction, registration score,
gap between the reading and the blank hypothesis, field kind.
Training data: synthetic pages (strategy 5) through the full pipeline with phone degradations (strategy 4);
never the specimen pages, which stay the test set.

python calibrate.py collect --model crnn.pt --omr omr.pt --pages 300 --out ~/dayone_local/cal_rows.jsonl
python calibrate.py fit --rows ~/dayone_local/cal_rows.jsonl --iso 0 --out ../../models/calibrator.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

W = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(W / "shared"), str(W / "strat2"), str(W / "strat4"), str(W / "strat5"), str(W / "strat7"),
                str(W / "strat11"), str(W / "strat18")]

KINDS = ["date", "date_or_year", "year", "bp", "int", "numunit", "enum_codes", "enum", "pattern", "free"]


def featurize(f: dict, kind: str) -> list[float]:
    x = [np.log(max(f["seqconf"], 1e-8)), f["minp"], min(f["n"], 40) / 40, float(f["snapped"]),
         np.clip(f["margin"], -20, 20) / 20, min(f["ink"] * 20, 1.0),
         float(np.clip(f["reg"], 0, 1)) if np.isfinite(f["reg"]) else 0.0, np.clip(f["gap"], -50, 50) / 50,
         float(f["n"] == 0), float(f.get("agree", 1.0))]
    x += [float(kind == k) for k in KINDS]
    return x


FEATURES = ["log_seqconf", "min_char_prob", "length", "snapped", "margin", "ink", "registration", "gap_vs_blank",
            "empty", "views_agree"] + [f"kind={k}" for k in KINDS]


class Calibrator:
    """Logistic calibrator (+ optional isotonic step) stored as plain JSON numbers, so the box does not depend on
    the scikit-learn version that fitted it (a pickled model can fail to load under another version)."""

    def __init__(self, coef, intercept, iso_x=None, iso_y=None):
        self.coef, self.intercept = np.asarray(coef, float), float(intercept)
        self.iso_x = None if iso_x is None else np.asarray(iso_x, float)
        self.iso_y = None if iso_y is None else np.asarray(iso_y, float)

    @staticmethod
    def from_sklearn(lr, iso=None):
        return Calibrator(lr.coef_[0], lr.intercept_[0], None if iso is None else iso.X_thresholds_,
                          None if iso is None else iso.y_thresholds_)

    def predict(self, feats: dict, kind: str = "free") -> float:
        z = float(np.dot(self.coef, featurize(feats, kind)) + self.intercept)
        p = 1.0 / (1.0 + np.exp(-z))
        if self.iso_x is not None:
            p = float(np.interp(p, self.iso_x, self.iso_y))      # = IsotonicRegression(out_of_bounds="clip")
        return float(p)

    def save(self, path):
        d = dict(coef=self.coef.tolist(), intercept=self.intercept,
                 iso_x=None if self.iso_x is None else self.iso_x.tolist(),
                 iso_y=None if self.iso_y is None else self.iso_y.tolist(), features=FEATURES)
        Path(path).write_text(json.dumps(d, indent=1), encoding="utf-8")

    @staticmethod
    def load(path):
        path = Path(path)
        if path.suffix == ".json":
            d = json.loads(path.read_text(encoding="utf-8"))
            return Calibrator(d["coef"], d["intercept"], d.get("iso_x"), d.get("iso_y"))
        import joblib                                   # legacy pickle (same scikit-learn version only)
        d = joblib.load(path)
        return Calibrator.from_sklearn(d["lr"], d.get("iso"))


def collect(model, omr, n_pages, out, seed=5000, holdout=True):
    from degrade import degrade_page
    from extract_zonal import Extractor
    from fieldlogic import kind
    from omr import OMR
    from recognizer import Recognizer
    from synth_pages import synth_page
    from vocab import canonical
    rec = Recognizer(model)
    ext = Extractor(rec, OMR(omr) if omr else None, None)
    rng = np.random.default_rng(seed)
    with open(out, "a", encoding="utf-8") as fo:
        for i in range(n_pages):
            t = int(rng.integers(1, 9))
            page, fields, meta = synth_page(t, rng, holdout_specimen=holdout)
            from status_bench import scribble            # crossed-out / scribbled values: any reading is wrong
            for f in fields:
                if f["type"] == "text" and f["value"] and f.get("bbox") and rng.random() < 0.06:
                    page = scribble(page, f["bbox"], rng, meta["ink"]); f["value"] = "<ILLISIBLE>"
            sev = int(rng.choice(5, p=[0.2, 0.25, 0.25, 0.18, 0.12]))
            photo, H, _ = degrade_page(page, sev, rng)
            pred = ext.extract(photo)
            pf = {f["key"]: f for f in pred["fields"]}
            for f in fields:
                if f["type"] != "text":
                    continue
                p = pf.get(f["key"])
                if p is None:
                    continue
                ok = canonical(p["value"]) == canonical(f["value"]) and pred["page_type"] == t
                fo.write(json.dumps(dict(feats=p["feats"], kind=kind(t, f["key"])["kind"], ok=bool(ok), sev=sev,
                                         gt_empty=f["value"] is None, lang=f.get("lang")), ensure_ascii=False) + "\n")
            if (i + 1) % 20 == 0:
                print(i + 1, flush=True)


def fit(rows_path, out, iso_on=True):
    import joblib
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    rows = [json.loads(l) for l in open(rows_path, encoding="utf-8")]
    X = np.array([featurize(r["feats"], r["kind"]) for r in rows])
    y = np.array([r["ok"] for r in rows], int)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(rows)); cut = int(0.7 * len(rows))
    tr, te = idx[:cut], idx[cut:]
    lr = LogisticRegression(C=1.0, max_iter=2000).fit(X[tr], y[tr])
    p_tr = lr.predict_proba(X[tr])[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip").fit(p_tr, y[tr])
    p_te = iso.predict(lr.predict_proba(X[te])[:, 1])
    sys.path.insert(0, str(W / "shared"))
    from eval import ece
    print(f"rows {len(rows)} acc {y.mean():.4f} | held-out ECE raw-seqconf "
          f"{ece(np.exp(X[te, 0]), y[te]):.4f} -> calibrated {ece(p_te, y[te]):.4f}")
    for tau in (0.8, 0.9, 0.95, 0.98, 0.99):
        m = p_te >= tau
        print(f"  tau {tau}: auto-accept {m.mean():.3f} of fields, accuracy {y[te][m].mean() if m.any() else float('nan'):.4f}")
    cal = Calibrator.from_sklearn(lr, iso if iso_on else None)
    if str(out).endswith(".json"):
        cal.save(out)
    else:
        joblib.dump(dict(lr=lr, iso=iso if iso_on else None), out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--model"); ap.add_argument("--omr"); ap.add_argument("--pages", type=int, default=200)
    ap.add_argument("--out"); ap.add_argument("--rows"); ap.add_argument("--seed", type=int, default=5000)
    ap.add_argument("--holdout", type=int, default=0)
    ap.add_argument("--iso", type=int, default=1)
    a = ap.parse_args()
    if a.cmd == "collect":
        collect(a.model, a.omr, a.pages, a.out, a.seed, bool(a.holdout))
    else:
        fit(a.rows, a.out, bool(a.iso))
