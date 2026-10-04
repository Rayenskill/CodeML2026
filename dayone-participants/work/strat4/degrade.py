"""Strategy 4: parametric field-photo degradations (seeded), keeping the page->photo homography.

degrade_page(img, severity, rng, families=None) -> (photo, H, meta)
  img      : clean page RGB uint8 (template resolution 1654x2339)
  severity : 0..4
  H        : 3x3 homography mapping clean-page pixel coords -> photo pixel coords
Families: perspective, rotation, background, blur, motion, noise, jpeg, downscale, light, shadow,
colour cast, curl-ish waviness, occlusion strip.
"""
from __future__ import annotations

import cv2
import numpy as np

FAMILIES = ["perspective", "blur", "motion", "noise", "jpeg", "light", "shadow", "cast", "occlusion", "wave"]


def _background(h, w, rng):
    kind = rng.integers(0, 3)
    base = rng.uniform(20, 120)
    noise = rng.normal(0, 1, (h // 8 + 1, w // 8 + 1)).astype(np.float32)
    noise = cv2.resize(noise, (w, h), interpolation=cv2.INTER_CUBIC)
    if kind == 0:      # fabric-like fine texture
        fine = rng.normal(0, 12, (h, w)).astype(np.float32)
        g = base + 10 * noise + fine
        col = np.stack([g, g * rng.uniform(0.9, 1.1), g * rng.uniform(0.9, 1.15)], -1)
    elif kind == 1:    # wood-like stripes
        x = np.linspace(0, rng.uniform(20, 60), w)[None, :] + 3 * noise
        g = base + 25 * np.sin(x) + 8 * noise
        col = np.stack([g * 0.6, g * 0.8, g * 1.1], -1)
    else:              # plain table
        g = base + 15 * noise
        col = np.stack([g, g, g], -1) * rng.uniform(0.8, 1.2, 3)
    return np.clip(col, 0, 255).astype(np.uint8)


def degrade_page(img: np.ndarray, severity: int, rng: np.random.Generator, families=None, out_long=1600):
    fam = set(FAMILIES if families is None else families)
    s = severity / 4.0
    h, w = img.shape[:2]
    meta = dict(severity=severity)
    # ---- geometry: page corners -> photo corners
    if "perspective" in fam and severity > 0:
        margin = rng.uniform(0.02, 0.10) * (0.3 + s)
        jit = 0.18 * s
        rot = np.deg2rad(rng.uniform(-12, 12) * s)
    else:
        margin, jit, rot = 0.0, 0.0, 0.0
    OH = out_long
    OW = int(round(out_long * w / h))
    scale = (1 - 2 * margin)
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    cx, cy = OW / 2, OH / 2
    dst = []
    for (x, y) in src:
        u = (x / w - 0.5) * OW * scale
        v = (y / h - 0.5) * OH * scale
        u, v = u * np.cos(rot) - v * np.sin(rot), u * np.sin(rot) + v * np.cos(rot)
        u += rng.uniform(-jit, jit) * OW * 0.5 * scale
        v += rng.uniform(-jit, jit) * OH * 0.5 * scale
        dst.append([cx + u, cy + v])
    dst = np.float32(dst)
    H = cv2.getPerspectiveTransform(src, dst)
    if margin > 0 or jit > 0 or rot != 0:
        bg = _background(OH, OW, rng)
        warped = cv2.warpPerspective(img, H, (OW, OH), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_CONSTANT)
        mask = cv2.warpPerspective(np.full((h, w), 255, np.uint8), H, (OW, OH))
        m = (mask.astype(np.float32) / 255)[..., None]
        out = (warped * m + bg * (1 - m)).astype(np.uint8)
    else:
        out = cv2.warpPerspective(img, H, (OW, OH), flags=cv2.INTER_AREA)
    out = out.astype(np.float32)
    # ---- waviness (paper not flat): small sinusoidal remap, not captured by H (residual error)
    if "wave" in fam and severity >= 2:
        amp = rng.uniform(1, 4) * s
        yy, xx = np.mgrid[0:OH, 0:OW].astype(np.float32)
        ph = rng.uniform(0, 6.28)
        mapx = xx + amp * np.sin(yy / OH * 6.28 * rng.uniform(0.5, 1.5) + ph)
        mapy = yy + amp * np.sin(xx / OW * 6.28 * rng.uniform(0.5, 1.5) + ph)
        out = cv2.remap(out, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    # ---- lighting
    if "light" in fam and severity > 0:
        gain = rng.uniform(1 - 0.55 * s, 1.0)
        gx = np.linspace(-1, 1, OW)[None, :] * rng.uniform(-1, 1)
        gy = np.linspace(-1, 1, OH)[:, None] * rng.uniform(-1, 1)
        grad = 1 + 0.35 * s * (gx + gy) / 2
        yy, xx = np.mgrid[0:OH, 0:OW]
        r = np.sqrt(((xx - OW / 2) / OW) ** 2 + ((yy - OH / 2) / OH) ** 2)
        vign = 1 - 0.5 * s * rng.uniform(0, 1) * r ** 2
        out = out * (gain * grad * vign)[..., None]
        gamma = rng.uniform(1.0, 1.0 + 0.6 * s)
        out = 255 * (np.clip(out, 0, 255) / 255) ** gamma
    if "shadow" in fam and severity >= 1 and rng.random() < 0.3 + 0.5 * s:
        mask = np.zeros((OH, OW), np.float32)
        pts = np.int32([[rng.uniform(0, OW), rng.uniform(0, OH)] for _ in range(rng.integers(3, 6))])
        cv2.fillPoly(mask, [cv2.convexHull(pts)], 1.0)
        k = int(rng.uniform(31, 151)) | 1
        mask = cv2.GaussianBlur(mask, (k, k), 0)
        out = out * (1 - rng.uniform(0.2, 0.6) * s * mask)[..., None]
    if "cast" in fam and severity > 0:
        out = out * rng.uniform(1 - 0.15 * s, 1 + 0.15 * s, 3)[None, None, :]
    if "occlusion" in fam and severity >= 3 and rng.random() < 0.25:
        # white paper strip (like 1-1.jpg) across a random band
        y0 = int(rng.uniform(0.1, 0.9) * OH); hh = int(rng.uniform(0.02, 0.05) * OH)
        x0 = int(rng.uniform(0, 0.5) * OW); ww = int(rng.uniform(0.2, 0.5) * OW)
        out[y0:y0 + hh, x0:x0 + ww] = rng.uniform(200, 250)
        meta["occlusion"] = [x0, y0, x0 + ww, y0 + hh]
    out = np.clip(out, 0, 255).astype(np.uint8)
    # ---- optics
    if "blur" in fam and severity > 0:
        sig = rng.uniform(0.3, 0.6 + 1.6 * s)
        out = cv2.GaussianBlur(out, (0, 0), sig)
    if "motion" in fam and severity >= 2 and rng.random() < 0.5:
        k = int(rng.uniform(3, 3 + 8 * s))
        ker = np.zeros((k, k), np.float32)
        ker[k // 2, :] = 1.0 / k
        M = cv2.getRotationMatrix2D((k / 2, k / 2), rng.uniform(0, 180), 1)
        ker = cv2.warpAffine(ker, M, (k, k))
        ker /= max(ker.sum(), 1e-6)
        out = cv2.filter2D(out, -1, ker)
    if "noise" in fam and severity > 0:
        out = np.clip(out.astype(np.float32) + rng.normal(0, 2 + 10 * s, out.shape), 0, 255).astype(np.uint8)
    if "jpeg" in fam and severity > 0:
        q = int(rng.uniform(95 - 60 * s, 95 - 30 * s))
        ok, enc = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, max(q, 25)])
        out = cv2.imdecode(enc, cv2.IMREAD_UNCHANGED)
    meta["H"] = H.tolist()
    return out, H, meta


def map_box(H, b):
    pts = np.float32([[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]])[None]
    q = cv2.perspectiveTransform(pts, H)[0]
    return [float(q[:, 0].min()), float(q[:, 1].min()), float(q[:, 0].max()), float(q[:, 1].max())]
