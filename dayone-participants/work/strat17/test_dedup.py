"""Tests for strategy 17 on the real data: 124 PNGs -> 80 unique pages; degraded re-shots of the same
page are near-duplicates; different patients' pages of the same type are not."""
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

W = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(W / "shared"), str(W / "strat2"), str(W / "strat4"), str(W / "strat17")]
from common import GT_DIR, REG  # noqa: E402
from dedup import content_similarity, diff_records, handwriting_hash, sha256_bytes  # noqa: E402


def test_exact_duplicates():
    rows = list(csv.DictReader(open(GT_DIR / "index.csv", encoding="utf-8")))
    by_hash = {}
    for r in rows:
        h = sha256_bytes((REG / r["png"]).read_bytes())
        by_hash.setdefault(h, set()).add(r["page"])
    assert len(rows) == 124
    assert len(by_hash) == 80 and all(len(v) == 1 for v in by_hash.values())


def near_duplicate_ink_study():
    from crops import blank_dark
    from degrade import degrade_page
    from register import register
    rng = np.random.default_rng(0)
    sims_same, sims_diff = [], []
    for t in (2, 3, 4):
        pages = [t + 8 * p for p in range(3)]
        hashes = {}
        for n in pages:
            img = cv2.cvtColor(cv2.imread(str(next(REG.glob(f"dossiers_specimen_10_patientes-{n:02d}*.png")))), cv2.COLOR_BGR2RGB)
            for k in range(2):
                ph, _, _ = degrade_page(img, 2, rng)
                r = register(ph, t)
                hashes[(n, k)] = handwriting_hash(r["warped"], blank_dark(t))
        for (n1, k1), h1 in hashes.items():
            for (n2, k2), h2 in hashes.items():
                if (n1, k1) < (n2, k2):
                    (sims_same if n1 == n2 else sims_diff).append(content_similarity(h1, h2))
    print("same page re-shot similarity min", min(sims_same), "different patients max", max(sims_diff))
    # NEGATIVE RESULT (2026-10-03): pre-OCR ink maps do not separate re-shots (min 0.24) from other
    # patients' pages of the same type (max 0.68) under severity-2 degradations. Near-duplicates are
    # therefore decided on extracted values (diff_records), see test_value_based_near_duplicates.
    return min(sims_same), max(sims_diff)


def test_diff_records():
    old = dict(fields=[dict(key="a", value="1"), dict(key="b", value=None)])
    assert diff_records(old, old)["verdict"] == "DOUBLON_SUSPECTÉ"
    new = dict(fields=[dict(key="a", value="1"), dict(key="b", value="12 SA")])
    d = diff_records(old, new)
    assert d["verdict"] == "MISE_À_JOUR" and d["new"] == [("b", "12 SA")]


if __name__ == "__main__":
    test_exact_duplicates(); print("exact ok")
    test_diff_records(); print("diff ok")
    print(near_duplicate_ink_study())


def test_value_based_near_duplicates(model=None):
    """Same page re-shot twice -> DOUBLON_SUSPECTÉ (most values identical); same page type of another
    patient -> MISE_À_JOUR-style diff with many changed values. Needs a trained recogniser."""
    import os
    model = model or os.environ.get("DAYONE_CRNN", str(Path.home() / "dayone_local" / "models" / "crnn_final.pt"))
    if not Path(model).exists():
        import pytest
        pytest.skip("no trained model")
    from degrade import degrade_page
    sys.path[:0] = [str(W / s) for s in ("strat7", "strat11", "strat18")]
    from extract_zonal import Extractor
    from recognizer import Recognizer
    ext = Extractor(Recognizer(model))
    rng = np.random.default_rng(1)

    def shot(n, t):
        img = cv2.cvtColor(cv2.imread(str(next(REG.glob(f"dossiers_specimen_10_patientes-{n:02d}*.png")))), cv2.COLOR_BGR2RGB)
        return ext.extract(degrade_page(img, 2, rng)[0], t)

    def agree(a, b):
        d = diff_records(a, b)
        filled = [k for k in d["same"]] + [c[0] for c in d["changed"]] + [x[0] for x in d["new"]] + [x[0] for x in d["removed"]]
        n_val = len(d["changed"]) + len(d["new"]) + len(d["removed"])
        same_filled = sum(1 for k in d["same"] if any(f["key"] == k and f.get("value") for f in a["fields"]))
        return same_filled / max(1, same_filled + n_val)
    same, diff = [], []
    for t in (2, 3, 6):
        a1, a2, b = shot(t, t), shot(t, t), shot(t + 8, t)
        same.append(agree(a1, a2)); diff.append(agree(a1, b))
    print("value agreement re-shot", same, "other patient", diff)
    assert min(same) > max(diff)
