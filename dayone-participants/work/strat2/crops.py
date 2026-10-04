"""Strategy 2 (part 3): zone crops from a registered (template-frame) page, shared by training and inference.

crop_zone(warped, t, key) -> RGB crop (zone + margins); tall zones are reduced to their ink band.
ink_score(warped, t, key) -> fraction of 'new ink' pixels (not explained by the blank form), used as a
blank / not-blank signal (NON_FOURNI detection) and as a confidence feature.
"""
from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

W = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(W / "shared"))
from common import LOCAL, PAGE_H, PAGE_W, SHARED  # noqa: E402

TEMPLATES = json.loads((SHARED / "templates.json").read_text(encoding="utf-8"))
MARGIN = (6, 8, 26, 8)            # left, top, right, bottom (px @200 dpi)
TALL = 110


@lru_cache(maxsize=None)
def blank_dark(t: int) -> np.ndarray:
    from register import ensure_templates
    ensure_templates()
    g = cv2.imread(str(LOCAL / "templates" / f"blank_p{t}.png"), cv2.IMREAD_GRAYSCALE).astype(np.float32)
    d = np.clip(1 - g / 235.0, 0, 1)
    return cv2.dilate(d, np.ones((7, 7), np.uint8))         # tolerate small misregistration


def darkness(rgb: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) + 1
    bg = cv2.dilate(g, np.ones((15, 15), np.uint8))
    bg = cv2.GaussianBlur(bg, (0, 0), 8)
    return np.clip(1 - g / bg, 0, 1)


def zone_box(t: int, key: str, margin=MARGIN):
    x0, y0, x1, y1 = TEMPLATES[str(t)]["zones"][key]["zone_px"]
    return (max(0, int(x0 - margin[0])), max(0, int(y0 - margin[1])),
            min(PAGE_W, int(x1 + margin[2])), min(PAGE_H, int(y1 + margin[3])))


def new_ink(warped: np.ndarray, t: int, box) -> np.ndarray:
    x0, y0, x1, y1 = box
    d = darkness(warped[y0:y1, x0:x1])
    tpl = blank_dark(t)[y0:y1, x0:x1]
    return np.clip(d - 1.2 * tpl, 0, 1)


def crop_zone(warped: np.ndarray, t: int, key: str):
    box = zone_box(t, key)
    x0, y0, x1, y1 = box
    if y1 - y0 > TALL:
        ink = new_ink(warped, t, box)
        prof = np.convolve((ink > 0.25).sum(1).astype(np.float32), np.ones(9) / 9, "same")
        if prof.max() > 3:
            c = int(np.argmax(np.convolve(prof, np.ones(50), "same")))
            y0, y1 = max(y0, y0 + c - 34), min(y1, y0 + c + 34)
    return warped[y0:y1, x0:x1], (x0, y0, x1, y1)


def ink_score(warped: np.ndarray, t: int, key: str, box=None) -> float:
    box = box or zone_box(t, key, (0, 0, 0, 0))
    ink = new_ink(warped, t, box)
    return float((ink > 0.3).mean())


def checkbox_crop(warped: np.ndarray, t: int, key: str, pad: int = 10):
    x0, y0, x1, y1 = TEMPLATES[str(t)]["zones"][key]["zone_px"]
    box = (int(x0 - pad), int(y0 - pad), int(x1 + pad), int(y1 + pad))
    return warped[box[1]:box[3], box[0]:box[2]], box
