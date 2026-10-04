"""Strategy 2 (part 2): register a photo of a registry page onto its template.

register(photo_rgb) -> dict(page_type, H (photo->template 3x3), warped (template-frame RGB),
                            score (ECC correlation), type_scores)
Steps: page quad (paper vs background) -> rectify to A4 -> page-type classification by ECC
correlation against the 8 blank templates -> ECC homography refinement (coarse to fine).
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
from common import LOCAL, PAGE_H, PAGE_W  # noqa: E402

TPL_DIR = LOCAL / "templates"


def ensure_templates():
    """Blank templates are derived from the specimen PDF (a few seconds); rebuilt automatically if missing."""
    if not (TPL_DIR / "blank_p8.png").exists():
        import templates
        templates.main()


@lru_cache(maxsize=None)
def template_gray(t: int, scale: float, sigma: float = 1.2) -> np.ndarray:
    ensure_templates()
    im = cv2.imread(str(TPL_DIR / f"blank_p{t}.png"), cv2.IMREAD_GRAYSCALE)
    w, h = int(PAGE_W * scale), int(PAGE_H * scale)
    im = cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)
    return _prep(im, sigma)


def _prep(g: np.ndarray, sigma: float = 1.2) -> np.ndarray:
    """Illumination-invariant 'ink darkness' map: 1 - gray / local paper level (shadows cancel)."""
    g = g.astype(np.float32) + 1
    k = max(9, (min(g.shape) // 25) | 1)
    bg = cv2.dilate(g, np.ones((k // 3 | 1, k // 3 | 1), np.uint8))       # local max = paper level
    bg = cv2.GaussianBlur(bg, (0, 0), k / 3)
    n = np.clip(1 - g / bg, 0, 1)
    if sigma > 0:
        n = cv2.GaussianBlur(n, (0, 0), sigma)
    return n


def order_quad(pts: np.ndarray) -> np.ndarray:
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(1); d = np.diff(pts, axis=1).ravel()
    tl, br = pts[np.argmin(s)], pts[np.argmax(s)]
    tr, bl = pts[np.argmin(d)], pts[np.argmax(d)]
    return np.float32([tl, tr, br, bl])


def find_page_quad(img: np.ndarray):
    """Return 4 corners (photo px) of the paper, or None if the page fills the frame."""
    h, w = img.shape[:2]
    sc = 800 / max(h, w)
    small = cv2.resize(img, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    v = cv2.GaussianBlur(hsv[..., 2], (7, 7), 0)
    _, m = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c) / (small.shape[0] * small.shape[1])
    if area < 0.2 or area > 0.97:
        return None
    hull = cv2.convexHull(c)
    peri = cv2.arcLength(hull, True)
    quad = None
    for eps in (0.01, 0.02, 0.03, 0.05):
        ap = cv2.approxPolyDP(hull, eps * peri, True)
        if len(ap) == 4:
            quad = ap
            break
    if quad is None:
        quad = cv2.boxPoints(cv2.minAreaRect(hull))
    return order_quad(np.asarray(quad, np.float32) / sc)


def _ecc(tpl, img, warp, motion=cv2.MOTION_HOMOGRAPHY, iters=60, eps=1e-5):
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iters, eps)
    try:
        cc, warp = cv2.findTransformECC(tpl, img, warp, motion, crit, None, 3)
    except cv2.error:
        return -1.0, warp
    if not np.isfinite(cc) or not np.all(np.isfinite(warp)):     # degenerate (flat) input: a failure, never a
        return -1.0, warp                                          # perfect score (inf would pass page_verdict)
    return float(cc), warp


def _scale_h(H, s):
    S = np.diag([s, s, 1.0]); Si = np.diag([1 / s, 1 / s, 1.0])
    return S @ H @ Si


def _refine(gray, H, t, s, motion, sigma, iters=60):
    S = np.diag([s, s, 1.0])
    ws = cv2.warpPerspective(gray, S @ H, (int(PAGE_W * s), int(PAGE_H * s)), flags=cv2.INTER_AREA,
                             borderMode=cv2.BORDER_REPLICATE)
    init = np.eye(3, dtype=np.float32) if motion == cv2.MOTION_HOMOGRAPHY else np.eye(2, 3, dtype=np.float32)
    cc, W = _ecc(template_gray(t, s, sigma), _prep(ws, sigma), init, motion, iters)
    if cc <= 0:
        return cc, H
    W = W.astype(np.float64)
    if W.shape[0] == 2:
        W = np.vstack([W, [0, 0, 1]])
    return cc, np.linalg.inv(S) @ np.linalg.inv(W) @ S @ H


def _coarse_scores(gray, H0, s=0.1, pad=12):
    """Shift-tolerant NCC of the rough rectified page against each blank template (page-type prior)."""
    S = np.diag([s, s, 1.0])
    W, Hh = int(PAGE_W * s), int(PAGE_H * s)
    r = _prep(cv2.warpPerspective(gray, S @ H0, (W, Hh), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE), 1.0)
    r = cv2.copyMakeBorder(r, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    out = {}
    for t in range(1, 9):
        tpl = template_gray(t, s, 1.0)[pad:-pad, pad:-pad]
        out[t] = float(cv2.matchTemplate(r, tpl, cv2.TM_CCOEFF_NORMED).max())
    return out


TWINS = {5: 7, 7: 5, 6: 8, 8: 6}


def _title_score(warped_gray, t):
    """NCC of the title band (PRÉCOCE vs TARDIF pages share the same layout)."""
    import json
    from common import SHARED
    global _TPL_JSON
    try:
        _TPL_JSON
    except NameError:
        _TPL_JSON = json.loads((SHARED / "templates.json").read_text(encoding="utf-8"))
    x0, y0, x1, y1 = _TPL_JSON[str(t)]["labels"][0]["bbox_px"]
    x0, y0, x1, y1 = int(x0) - 10, int(y0) - 10, int(x1) + 10, int(y1) + 10
    a = _prep(warped_gray[y0:y1, x0:x1], 1.0)
    b = template_gray(t, 1.0, 1.0)[y0:y1, x0:x1]
    a = cv2.copyMakeBorder(a, 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=0)
    return float(cv2.matchTemplate(a, b, cv2.TM_CCOEFF_NORMED).max())


@lru_cache(maxsize=None)
def _twin_pair(t: int, s: float = 0.5):
    """Template of t, template of its twin aligned onto t, and the mask where they truly differ."""
    o = TWINS[t]
    dt, do = template_gray(t, s, 1.0), template_gray(o, s, 1.0)
    cc, Wm = _ecc(dt, do, np.eye(2, 3, dtype=np.float32), cv2.MOTION_AFFINE, 100)
    do_al = cv2.warpAffine(do, Wm, (dt.shape[1], dt.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    m = (np.abs(dt - do_al) > 0.15).astype(np.uint8)
    m[: int(0.02 * m.shape[0])] = 0
    m = cv2.dilate(m, np.ones((5, 5), np.uint8)).astype(bool)
    return dt, do_al, m


def twin_decide(warped: np.ndarray, t: int, s: float = 0.5) -> int:
    """Pages 5/7 and 6/8 share the layout: compare only where their (aligned) blank forms differ."""
    o = TWINS[t]
    g = cv2.resize(cv2.cvtColor(warped, cv2.COLOR_RGB2GRAY), (int(PAGE_W * s), int(PAGE_H * s)), interpolation=cv2.INTER_AREA)
    d = _prep(g, 1.0)
    dt, do, m = _twin_pair(t, s)

    def ncc(a, b):
        a = a[m] - a[m].mean(); b = b[m] - b[m].mean()
        return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-9))
    return t if ncc(d, dt) >= ncc(d, do) else o


def _full_refine(gray, H0, t, fine_scale):
    cc, H = _refine(gray, H0, t, 0.15, cv2.MOTION_AFFINE, 2.0, 40)
    if cc <= 0:
        H = H0
    for s, sig in ((0.2, 1.5), (0.35, 1.2), (fine_scale, 1.0)):
        cc2, H2 = _refine(gray, H, t, s, cv2.MOTION_HOMOGRAPHY, sig, 60)
        if cc2 >= cc - 0.03:
            H, cc = H2, cc2
    return cc, H


def register(photo: np.ndarray, page_type: int | None = None, fine_scale: float = 0.5, topk: int = 2):
    h, w = photo.shape[:2]
    quad = find_page_quad(photo)
    if quad is None:
        quad = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[0, 0], [PAGE_W, 0], [PAGE_W, PAGE_H], [0, PAGE_H]])
    H0 = cv2.getPerspectiveTransform(quad, dst).astype(np.float64)        # photo -> template (rough)
    gray = cv2.cvtColor(photo, cv2.COLOR_RGB2GRAY)
    prior = _coarse_scores(gray, H0)
    if page_type:
        cands = [page_type]
    else:
        cands = sorted(prior, key=prior.get, reverse=True)[:topk]
        for t in list(cands):
            if t in TWINS and TWINS[t] not in cands:
                cands.append(TWINS[t])
    res = {}
    for t in cands:
        res[t] = _full_refine(gray, H0, t, fine_scale)
    # twins share the geometry: decide on the full-page cc, then on the title band
    best = max(res, key=lambda t: res[t][0])
    cc, H = res[best]
    warped = cv2.warpPerspective(photo, H, (PAGE_W, PAGE_H), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)
    if not page_type and best in TWINS:
        best = twin_decide(warped, best)
    return dict(page_type=best, H=H, warped=warped, score=float(cc),
                type_scores={int(k): float(v[0]) for k, v in res.items()}, prior=prior)


ROTATIONS = (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180)


def register_any_orientation(photo: np.ndarray, page_type: int | None = None, accept: float = 0.45,
                             good: float = 0.70):
    """register(); if the page is not found upright (score < accept), try the photo turned by 90°, 270° and 180°
    (phone held sideways, booklet upside down). Returns (reg, photo as registered, rotation code or None)."""
    reg = register(photo, page_type)
    best = (reg, photo, None)
    if reg["score"] >= accept:
        return best
    for rot in ROTATIONS:
        turned = cv2.rotate(photo, rot)
        r = register(turned, page_type)
        if r["score"] > best[0]["score"]:
            best = (r, turned, rot)
        if r["score"] >= good:
            break
    return best
