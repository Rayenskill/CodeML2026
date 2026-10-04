"""Strategy 17: document assembly helpers: exact / near-duplicate detection and re-scan diffing.

* exact duplicate  : SHA-256 of the uploaded bytes (double tap, retry)
* near duplicate   : perceptual hash (DCT pHash 64-bit) of the *registered, identifier-masked* page
                     -> same page photographed again; Hamming distance threshold
* re-scan diff     : compare extracted values of two captures of the same page type & booklet:
                     identical content -> DOUBLON_SUSPECTÉ ; new values -> update proposal
"""
from __future__ import annotations

import hashlib

import cv2
import numpy as np


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def phash(img_rgb: np.ndarray, size=32, keep=8) -> int:
    g = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY) if img_rgb.ndim == 3 else img_rgb
    g = cv2.resize(g, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)
    d = cv2.dct(g)[:keep, :keep]
    med = np.median(d[1:].ravel()) if keep > 1 else d.mean()
    bits = (d > med).ravel()
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def handwriting_hash(warped_rgb: np.ndarray, blank_dark: np.ndarray, size=(165, 234)) -> np.ndarray:
    """Map of handwriting ink (registered page minus blank form, binarised, blurred) -> compares content."""
    g = cv2.cvtColor(warped_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) + 1
    bg = cv2.GaussianBlur(cv2.dilate(g, np.ones((15, 15), np.uint8)), (0, 0), 8)
    ink = (np.clip(1 - g / bg - 1.2 * blank_dark, 0, 1) > 0.25).astype(np.float32)
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    small = cv2.resize(ink, size, interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(small, (0, 0), 1.0).ravel()


def content_similarity(h1: np.ndarray, h2: np.ndarray) -> float:
    a, b = h1 - h1.mean(), h2 - h2.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-9))


def diff_records(old: dict, new: dict):
    """Field-level comparison of two extractions of the same page (re-digitisation, brief task 7)."""
    of = {f["key"]: f for f in old["fields"]}
    out = dict(same=[], new=[], changed=[], removed=[])
    for f in new["fields"]:
        o = of.get(f["key"])
        ov, nv = (o or {}).get("value"), f.get("value")
        if ov == nv:
            out["same"].append(f["key"])
        elif not ov and nv:
            out["new"].append((f["key"], nv))
        elif ov and not nv:
            out["removed"].append((f["key"], ov))
        else:
            out["changed"].append((f["key"], ov, nv))
    filled = [k for k in out["same"] if (of.get(k) or {}).get("value")]
    out["verdict"] = "DOUBLON_SUSPECTÉ" if not out["new"] and not out["changed"] and filled else "MISE_À_JOUR"
    return out
