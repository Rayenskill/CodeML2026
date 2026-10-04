"""Strategy 11: build the recogniser training set from synthetic pages pushed through the real pipeline.

synthetic page (strategy 5) -> phone-photo degradation (strategy 4) -> back to template frame with a
simulated registration error -> zone crops (strategy 2 crops.py) -> height-normalised crops + labels.
Also writes checkbox crops for the OMR model (strategy 18).

python gen_dataset.py --pages 3000 --out ~/dayone_local/ds_v1 --workers 16
"""
from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

W = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(W / "shared"), str(W / "strat2"), str(W / "strat4"), str(W / "strat5")]
from common import PAGE_H, PAGE_W  # noqa: E402

CROP_H = 64
TYPE_P = np.array([1, 2, 4, 1.2, 1.2, 1.2, 1.2, 1.2], float)
TYPE_P /= TYPE_P.sum()
SEV_P = (0.2, 0.25, 0.25, 0.18, 0.12)


def perturb(rng, sigma):
    src = np.float32([[0, 0], [PAGE_W, 0], [PAGE_W, PAGE_H], [0, PAGE_H]])
    dst = src + rng.normal(0, sigma, src.shape).astype(np.float32)
    return cv2.getPerspectiveTransform(src, dst)


def resize_h(img, h=CROP_H):
    sc = h / img.shape[0]
    w = max(8, int(round(img.shape[1] * sc)))
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA if sc < 1 else cv2.INTER_LINEAR)


_BANKS = {}


def _bank(path):
    """Strategy 21 handwriting bank (verified generated patches), loaded once per worker."""
    if path not in _BANKS:
        sys.path.insert(0, str(W / "strat21"))
        from gemini_handwriting import load_bank
        _BANKS[path] = load_bank(Path(path))
    return _BANKS[path]


def work(args):
    idx, out, seed, holdout = args[:4]
    cb_only = len(args) > 4 and args[4]
    hw_aug = args[5] if len(args) > 5 else 0.0
    sev_p = args[6] if len(args) > 6 else SEV_P
    bank_dir, bank_p = (args[7], args[8]) if len(args) > 8 else (None, 0.0)
    from crops import checkbox_crop, crop_zone, ink_score
    from degrade import degrade_page
    from synth_pages import synth_page
    rng = np.random.default_rng(seed)
    t = int(rng.choice(np.arange(1, 9), p=TYPE_P))
    bank = _bank(bank_dir) if bank_dir and bank_p > 0 else None
    page, fields, meta = synth_page(t, rng, holdout_specimen=holdout, hw_aug=hw_aug, bank=bank, bank_p=bank_p)
    sev = int(rng.choice(5, p=np.asarray(sev_p) / np.sum(sev_p)))
    photo, H, _ = degrade_page(page, sev, rng)
    Hback = perturb(rng, 1.5 + sev) @ np.linalg.inv(H)
    warped = cv2.warpPerspective(photo, Hback, (PAGE_W, PAGE_H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    rows, cbrows = [], []
    for j, f in enumerate(fields):
        if f["type"] == "identifier":
            continue
        if f["type"] == "checkbox":
            if rng.random() > (1.0 if cb_only else 0.35):
                continue
            c, _ = checkbox_crop(warped, t, f["key"])
            c = cv2.resize(c, (40, 40), interpolation=cv2.INTER_AREA)
            buf = cv2.imencode(".png", cv2.cvtColor(c, cv2.COLOR_RGB2BGR))[1].tobytes()
            cbrows.append(dict(path=f"{idx:06d}_{j:03d}_cb", label=int(f["value"]), t=t, key=f["key"], sev=sev, _b=buf))
            continue
        if cb_only:
            continue
        empty = f["value"] is None
        if empty and rng.random() > 0.12:
            continue
        crop, box = crop_zone(warped, t, f["key"])
        if crop.size == 0:
            continue
        im = resize_h(crop)
        buf = cv2.imencode(".jpg", cv2.cvtColor(im, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes()
        rows.append(dict(path=f"{idx:06d}_{j:03d}", _b=buf, w=int(im.shape[1]), text=f["value"] or "", status=f["status"], t=t, key=f["key"],
                         lang=f.get("lang"), font=f.get("font") or meta["font"], sev=sev, printed=meta["printed"],
                         ink=round(ink_score(warped, t, f["key"]), 4), writer=meta.get("writer", False)))
    return rows, cbrows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=int, default=2000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--holdout", type=int, default=1, help="1 = exclude the 5 specimen fonts")
    ap.add_argument("--cb_only", type=int, default=0)
    ap.add_argument("--hw_aug", type=float, default=0.0, help="share of pages written by an irregular 'writer'")
    ap.add_argument("--sev_p", type=float, nargs=5, default=list(SEV_P), help="probabilities of severities 0..4")
    ap.add_argument("--bank", default=None, help="strategy 21 handwriting bank directory")
    ap.add_argument("--bank_p", type=float, default=0.0, help="share of filled fields drawn from the bank")
    a = ap.parse_args()
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(i, str(out), a.seed + i, bool(a.holdout), bool(a.cb_only), a.hw_aug, tuple(a.sev_p), a.bank, a.bank_p) for i in range(a.pages)]
    n = 0
    # packed output (one binary file + offsets): Windows is very slow at opening many small files
    with open(out / "packed.jsonl", "a", encoding="utf-8") as fl, open(out / "cb_packed.jsonl", "a", encoding="utf-8") as fc, \
            open(out / "crops.bin", "ab") as bt, open(out / "cb.bin", "ab") as bc, Pool(a.workers) as pool:
        off_t, off_c = bt.tell(), bc.tell()
        for rows, cbrows in pool.imap_unordered(work, jobs, chunksize=4):
            for r in rows:
                b = r.pop("_b"); bt.write(b); r.update(off=off_t, len=len(b)); off_t += len(b)
                fl.write(json.dumps(r, ensure_ascii=False) + "\n")
            for r in cbrows:
                b = r.pop("_b"); bc.write(b); r.update(off=off_c, len=len(b)); off_c += len(b)
                fc.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
            if n % 100 == 0:
                print(n, flush=True)


if __name__ == "__main__":
    main()
