"""Sideways / upside-down captures are registered by turning the photo back (strategy 2 + 4)."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

W = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(W / "shared"), str(W / "strat2"), str(W / "strat4")]
from common import GT_DIR, REG  # noqa: E402
from degrade import degrade_page  # noqa: E402
from register import register_any_orientation  # noqa: E402


def _photo(n=3, sev=1):
    g = json.loads((GT_DIR / f"page_{n:02d}.json").read_text(encoding="utf-8"))
    img = cv2.cvtColor(cv2.imread(str(REG / g["png"][0])), cv2.COLOR_BGR2RGB)
    return degrade_page(img, sev, np.random.default_rng(7))[0], g["page_type"]


def test_upright_photo_is_not_turned():
    photo, t = _photo()
    reg, _, rot = register_any_orientation(photo)
    assert rot is None and reg["page_type"] == t and reg["score"] > 0.7


def test_sideways_and_upside_down_photos_are_recovered():
    photo, t = _photo()
    for turn, undo in ((cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE),
                       (cv2.ROTATE_180, cv2.ROTATE_180)):
        reg, used, rot = register_any_orientation(cv2.rotate(photo, turn))
        assert rot == undo and reg["page_type"] == t and reg["score"] > 0.7, (turn, rot, reg["score"])
        assert used.shape == photo.shape


def test_other_registry_model_still_rejected():
    real = cv2.cvtColor(cv2.imread(str(REG / "1-3.jpg")), cv2.COLOR_BGR2RGB)
    reg, _, _ = register_any_orientation(real)
    assert reg["score"] < 0.45


def test_degenerate_photo_never_scores_as_registered():
    """A flat / blank photo once produced an infinite ECC score, which passed the page verdict."""
    from register import register
    for img in (np.full((1600, 1131, 3), 200, np.uint8), np.zeros((1600, 1131, 3), np.uint8)):
        reg = register(img)
        assert np.isfinite(reg["score"]) and reg["score"] < 0.45, reg["score"]
