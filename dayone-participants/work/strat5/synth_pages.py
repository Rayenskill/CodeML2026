"""Strategy 5: synthetic filled registry pages (FR / EN / AR / mixed, handwriting + print) with exact GT.

synth_page(t, rng, fonts) -> (rgb page in template frame 1654x2339, fields)
fields: [{key, type: text|checkbox|identifier, value, status, bbox (px), lang, font}]
Values are drawn into the template zones like the organisers' generator (value glyphs missing from
a font are left blank, exactly as in the specimen pages written with NanumPen/Gaegu).
"""
from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

W = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(W / "shared"), str(W / "strat5")]
from common import GT_DIR, LOCAL, SHARED  # noqa: E402
from vocab import infer_kinds, sample_value  # noqa: E402

try:
    import arabic_reshaper
    from bidi.algorithm import get_display
except ImportError:  # pragma: no cover
    arabic_reshaper = None

FONT_DIR = LOCAL / "fonts"
SPECIMEN_FONTS = {"Caveat[wght].ttf", "ShadowsIntoLight.ttf", "NanumPenScript-Regular.ttf", "Gaegu-Regular.ttf",
                  "ReenieBeanie.ttf"}
PRINTED = [r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\calibri.ttf", r"C:\Windows\Fonts\times.ttf",
           r"C:\Windows\Fonts\verdana.ttf", r"C:\Windows\Fonts\tahoma.ttf", r"C:\Windows\Fonts\segoeui.ttf",
           r"C:\Windows\Fonts\georgia.ttf", r"C:\Windows\Fonts\cour.ttf"]
PRINTED = [p for p in PRINTED if Path(p).exists()]
INKS = [(20, 33, 140), (25, 40, 120), (15, 20, 90), (30, 30, 35), (10, 10, 20), (40, 60, 160), (60, 25, 110)]
FAKE_IDS = dict(nom=["Tazi Meryem", "Saidi Fatima", "Alaoui Khadija", "Benali Aicha", "Idrissi Salma", "فاطمة الزهراء"],
                cin=["CB609814", "BK941500", "AB123456", "EE778812", "J455091"],
                tel=["06 00 76 13 48", "06 61 22 90 14", "07 12 34 56 78"],
                adr=["49 Rue Al Qods, Kénitra", "Douar Ait Ali, Azilal", "12 Hay Salam, Meknès", "حي السلام، مكناس"])

TEMPLATES = json.loads((SHARED / "templates.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def coverage():
    from fontTools.ttLib import TTFont
    out = {}
    fonts = sorted(FONT_DIR.glob("*.ttf")) + [Path(p) for p in PRINTED]
    for f in fonts:
        try:
            cm = TTFont(str(f), fontNumber=0, lazy=True).getBestCmap() or {}
        except Exception:
            continue
        out[str(f)] = frozenset(cm)
    return out


def font_lists(holdout_specimen: bool):
    cov = coverage()
    latin, arabic = [], []
    for f, cm in cov.items():
        name = Path(f).name
        if holdout_specimen and name in SPECIMEN_FONTS:
            continue
        if f in PRINTED:
            continue
        if all(ord(c) in cm for c in "aeAZ09/"):
            latin.append(f)
        if all(ord(c) in cm for c in "ابتسلمي"):
            arabic.append(f)
    return latin, arabic


@lru_cache(maxsize=4096)
def _font(path, size):
    return ImageFont.truetype(path, size)


@lru_cache(maxsize=512)
def _digit_height(path):
    f = _font(path, 100)
    b = f.getbbox("0123456789")
    return max(b[3] - b[1], 1)


@lru_cache(maxsize=None)
def stats():
    """Fill rate per (page_type, key) and tick rate per checkbox from the specimen GT."""
    fill, tick = {}, {}
    cnt = {}
    for f in sorted(GT_DIR.glob("page_*.json")):
        g = json.loads(f.read_text(encoding="utf-8"))
        for x in g["fields"]:
            k = (g["page_type"], x["key"])
            cnt[k] = cnt.get(k, 0) + 1
            if x["type"] == "checkbox":
                tick[k] = tick.get(k, 0) + int(x["value"])
            else:
                fill[k] = fill.get(k, 0) + int(x["status"] == "CONNU" or bool(x.get("identifier")))
    return {k: v / cnt[k] for k, v in fill.items()}, {k: v / cnt[k] for k, v in tick.items()}


@lru_cache(maxsize=None)
def kinds():
    return infer_kinds()[0]


def kind_for(t, key):
    k = kinds()
    if t == 3 and key.startswith("visites."):
        return k.get((t, "visites." + key.split(".")[1]), dict(kind="free"))
    return k.get((t, key), dict(kind="free"))


def is_arabic(s):
    return any("\u0600" <= c <= "\u06ff" for c in s)


def visual(s):
    """String as drawn left-to-right (Arabic shaped and reordered)."""
    if is_arabic(s) and arabic_reshaper is not None:
        return get_display(arabic_reshaper.reshape(s))
    return s


@lru_cache(maxsize=None)
def _archives():
    import pymupdf as fitz
    return fitz.Archive(str(FONT_DIR)), fitz.Archive(r"C:\Windows\Fonts")


def render_text_shaped(text, font_path, px_height, ink, rng):
    """Arabic / mixed text with proper shaping (HarfBuzz via PyMuPDF). Returns RGBA patch."""
    import pymupdf as fitz
    fa, fw = _archives()
    name = Path(font_path).name
    arch = fw if str(font_path).lower().startswith(r"c:\windows") else fa
    size = px_height * 1.55
    width = max(60, int(len(text) * size * 0.9 + 40))
    doc = fitz.open()
    page = doc.new_page(width=width, height=int(size * 2.4))
    col = "#%02x%02x%02x" % tuple(ink)
    css = f"@font-face {{font-family: F; src: url({name});}} * {{font-family: F; font-size: {size:.1f}px; color: {col}; white-space: nowrap;}}"
    page.insert_htmlbox(fitz.Rect(5, 5, width - 5, size * 2.4 - 5), f'<p dir="auto">{text}</p>', css=css, archive=arch)
    pix = page.get_pixmap(dpi=72, alpha=True)
    im = Image.frombytes("RGBA", (pix.width, pix.height), pix.samples)
    a = np.asarray(im)[..., 3]
    ys, xs = np.nonzero(a > 10)
    if len(xs) == 0:
        return Image.new("RGBA", (20, int(px_height) + 8), (0, 0, 0, 0))
    im = im.crop((max(xs.min() - 4, 0), max(ys.min() - 4, 0), xs.max() + 5, ys.max() + 5))
    alpha = rng.uniform(0.8, 1.0)
    arr = np.asarray(im).copy()
    arr[..., 3] = (arr[..., 3] * alpha).astype(np.uint8)
    im = Image.fromarray(arr)
    ang = rng.uniform(-2.5, 2.5)
    if abs(ang) > 0.3:
        im = im.rotate(ang, resample=Image.BICUBIC, expand=True)
    return im


def writer_style(rng):
    """One 'writer' per page: how this midwife deviates from the font (slant, spacing, irregularity, pen).
    Fonts are regular; real handwriting is not, so the recogniser must not learn a font's exact geometry."""
    return dict(shear=rng.uniform(-0.35, 0.25) if rng.random() < 0.6 else 0.0,
                track=rng.uniform(-0.06, 0.22),            # extra letter spacing, x font size
                wander=rng.uniform(0.0, 0.08),             # smooth baseline wander amplitude, x font size
                wave=rng.uniform(2.5, 8.0),                # its wavelength, in letters
                base_sd=rng.uniform(0.0, 0.025),           # small independent jitter per letter, x font size
                size_sd=rng.uniform(0.0, 0.06),            # per-letter size irregularity
                xscale=rng.uniform(0.8, 1.2),
                pen=int(rng.choice([-1, 0, 0, 1, 2])),     # thinner / same / thicker strokes
                elastic=rng.uniform(0.0, 2.2))             # smooth random displacement (px)


def _charwise(vis, font_path, size, ink, alpha, rng, style):
    """Letter-by-letter rendering on a common baseline with irregular size, spacing and baseline."""
    n = len(vis)
    W0 = int(size * (n + 2) * 1.6) + 20
    H0 = int(size * 2.6) + 20
    patch = Image.new("RGBA", (W0, H0), (0, 0, 0, 0))
    d = ImageDraw.Draw(patch)
    x, base = 10.0, H0 * 0.68
    ph = rng.uniform(0, 2 * np.pi)
    for i, c in enumerate(vis):
        s = max(6, int(round(size * (1 + rng.normal(0, style["size_sd"])))))
        f = _font(font_path, s)
        y = base + size * (style["wander"] * np.sin(2 * np.pi * i / style["wave"] + ph)
                           + rng.normal(0, style["base_sd"]))
        if c != " ":
            d.text((x, y), c, font=f, fill=tuple(ink) + (alpha,), anchor="ls")
        x += f.getlength(c) + style["track"] * size * rng.uniform(0.5, 1.5)
    return patch


def handwrite_aug(patch, rng, style):
    """Shear, horizontal scale, pen width and elastic distortion of a rendered RGBA patch."""
    import cv2
    a = np.asarray(patch).copy()
    h, w = a.shape[:2]
    pad = int(0.4 * h) + 4
    a = cv2.copyMakeBorder(a, 4, 4, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
    h, w = a.shape[:2]
    sh = style["shear"] + rng.normal(0, 0.04)
    sx = style["xscale"] * rng.uniform(0.95, 1.05)
    M = np.float32([[sx, -sh, sh * h * 0.6], [0, 1, 0]])
    a = cv2.warpAffine(a, M, (int(w * sx) + 4, h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))
    if style["elastic"] > 0.2:
        hh, ww = a.shape[:2]
        sig = max(4.0, 0.35 * hh)
        dx = cv2.GaussianBlur(rng.normal(0, 1, (hh, ww)).astype(np.float32), (0, 0), sig)
        dy = cv2.GaussianBlur(rng.normal(0, 1, (hh, ww)).astype(np.float32), (0, 0), sig)
        k = style["elastic"] / max(1e-6, float(np.abs(np.concatenate([dx, dy])).max()))
        yy, xx = np.mgrid[0:hh, 0:ww].astype(np.float32)
        a = cv2.remap(a, xx + dx * k, yy + dy * k, cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))
    if style["pen"]:
        al = a[..., 3]
        ker = np.ones((2, 2), np.uint8) if abs(style["pen"]) == 1 else np.ones((3, 3), np.uint8)
        if style["pen"] > 0:
            al2 = cv2.dilate(al, ker)
        else:
            al2 = cv2.erode(al, ker)
            if (al2 > 60).sum() < 0.55 * max(1, (al > 60).sum()):    # thin font: do not erase strokes
                al2 = al
        a[..., 3] = al2
        ink_rgb = a[al > 60][:, :3]
        if len(ink_rgb):
            a[al2 > 0, :3] = np.median(ink_rgb, 0).astype(np.uint8)
    ys, xs = np.nonzero(a[..., 3] > 10)
    if len(xs) == 0:
        return patch
    return Image.fromarray(a[max(0, ys.min() - 4):ys.max() + 5, max(0, xs.min() - 4):xs.max() + 5])


def render_text(text, font_path, px_height, ink, rng, cov, style=None):
    """Render text into an RGBA patch; unsupported glyphs are left blank. Returns patch.
    style (writer_style) adds handwriting irregularity on top of the font."""
    if is_arabic(text):
        patch = render_text_shaped(text, font_path, px_height, ink, rng)
        return handwrite_aug(patch, rng, style) if style else patch
    dh = _digit_height(font_path)
    size = max(8, int(round(100 * px_height / dh)))
    font = _font(font_path, size)
    drawn = "".join(c if (ord(c) in cov or c == " ") else " " for c in text)
    vis = visual(drawn)
    alpha = int(rng.uniform(200, 255))
    if style:
        patch = _charwise(vis, font_path, size, ink, alpha, rng, style)
        a = np.asarray(patch)[..., 3]
        ys, xs = np.nonzero(a > 10)
        if len(xs) == 0:
            return Image.new("RGBA", (20, int(px_height) + 8), (0, 0, 0, 0))
        patch = patch.crop((max(0, xs.min() - 4), max(0, ys.min() - 4), xs.max() + 5, ys.max() + 5))
        patch = handwrite_aug(patch, rng, style)
    else:
        b = font.getbbox(vis)
        w, h = b[2] - b[0] + 8, b[3] - b[1] + 8
        if w <= 8:
            w = 20
        patch = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(patch)
        d.text((4 - b[0], 4 - b[1]), vis, font=font, fill=tuple(ink) + (alpha,))
    ang = rng.uniform(-2.5, 2.5)
    if abs(ang) > 0.3:
        patch = patch.rotate(ang, resample=Image.BICUBIC, expand=True)
    return patch


def draw_tick(d, box, ink, rng):
    """Mark styles seen on registries: X, check mark, filled, hatching, single slash, scribble, circle."""
    x0, y0, x1, y1 = box
    w = x1 - x0
    o = rng.uniform(-0.1, 0.25) * w
    lw = int(rng.integers(2, 5))
    kind = rng.choice(["x", "check", "fill", "hatch", "slash", "scribble", "circle"],
                      p=[0.35, 0.2, 0.07, 0.18, 0.08, 0.07, 0.05])
    if kind == "x":
        d.line([(x0 - o, y0 - o), (x1 + o, y1 + o)], fill=ink, width=lw)
        d.line([(x0 - o, y1 + o), (x1 + o, y0 - o)], fill=ink, width=lw)
    elif kind == "check":
        d.line([(x0, y0 + 0.55 * w), (x0 + 0.4 * w, y1 + o)], fill=ink, width=lw)
        d.line([(x0 + 0.4 * w, y1 + o), (x1 + 2 * o, y0 - 2 * o)], fill=ink, width=lw)
    elif kind == "fill":
        d.rectangle([x0 + 2, y0 + 2, x1 - 2, y1 - 2], fill=ink)
    elif kind == "hatch":
        n = int(rng.integers(3, 6))
        sl = rng.uniform(0.15, 0.45) * w
        for k in range(n):
            yy = y0 + (k + 0.5) * w / n + rng.uniform(-1.5, 1.5)
            d.line([(x0 - o, yy + sl / 2), (x1 + o, yy - sl / 2)], fill=ink, width=max(2, lw - 1))
    elif kind == "slash":
        d.line([(x0 - o, y1 + o), (x1 + o, y0 - o)], fill=ink, width=lw)
    elif kind == "scribble":
        pts = [(x0 + rng.uniform(0, w), y0 + rng.uniform(0, w)) for _ in range(int(rng.integers(5, 9)))]
        d.line(pts, fill=ink, width=max(2, lw - 1))
    else:
        r = 0.9 * w
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=ink, width=lw)


@lru_cache(maxsize=8)
def blank(t):
    return Image.open(LOCAL / "templates" / f"blank_p{t}.png").convert("RGB")


def bank_patch(entry, px_height, ink):
    """RGBA handwriting patch from the bank, rescaled so its ink band is ~1.6x the digit height, in the page ink."""
    im = Image.open(entry["path"]).convert("RGBA")
    sc = 1.6 * px_height / max(1, im.height)
    im = im.resize((max(4, int(im.width * sc)), max(4, int(im.height * sc))), Image.LANCZOS)
    a = np.asarray(im).copy()
    a[..., :3] = np.asarray(ink, np.uint8)
    return Image.fromarray(a, "RGBA")


def synth_page(t: int, rng: np.random.Generator, holdout_specimen=True, lang=None, printed=None, hw_aug=0.0,
               bank=None, bank_p=0.0):
    """hw_aug: probability that the page is written by an irregular 'writer' (writer_style) instead of the
    font's exact geometry. bank ({kind: [entry]}, strategy 21) + bank_p: share of filled fields drawn from a bank
    of verified handwriting patches (e.g. generated by an image model) instead of a font."""
    latin, arabic = font_lists(holdout_specimen)
    cov = coverage()
    fill_rate, tick_rate = stats()
    page = blank(t).copy()
    # small global paper tint variation
    if rng.random() < 0.5:
        arr = np.asarray(page).astype(np.float32) * rng.uniform(0.93, 1.05, 3)
        page = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(page)
    lang = lang or rng.choice(["fr", "en", "ar", "mix"], p=[0.55, 0.15, 0.15, 0.15])
    printed = (rng.random() < 0.15) if printed is None else printed
    main_font = str(rng.choice(PRINTED if printed else latin))
    ar_font = str(rng.choice(arabic))
    ink = tuple(int(c) for c in (INKS[rng.integers(len(INKS))] if not printed else (20, 20, 25)))
    base_h = rng.uniform(12.5, 30)         # digit height in px (specimen ink: p10 13-16, median 18-27)
    style = writer_style(rng) if (not printed and hw_aug and rng.random() < hw_aug) else None
    fields = []
    zones = TEMPLATES[str(t)]["zones"]
    for key, z in zones.items():
        x0, y0, x1, y1 = z["zone_px"]
        if z["type"] == "checkbox":
            p = 0.15 + 0.7 * tick_rate.get((t, key), 0.3)
            val = bool(rng.random() < p)
            if val:
                draw_tick(d, (x0, y0, x1, y1), ink if not printed else (20, 20, 25), rng)
            fields.append(dict(key=key, type="checkbox", value=val, status="CONNU", bbox=[x0, y0, x1, y1]))
            continue
        ident = key.startswith("inline.") and any(s in key for s in ("parturiente", "patiente", "cin", "adresse",
                                                                    "telephone", "nom_du_mari"))
        fl = lang if lang != "mix" else rng.choice(["fr", "en", "ar"], p=[0.5, 0.25, 0.25])
        if ident:
            pool = FAKE_IDS["cin"] if "cin" in key else FAKE_IDS["tel"] if "telephone" in key else \
                FAKE_IDS["adr"] if "adresse" in key else FAKE_IDS["nom"]
            value = str(rng.choice(pool)); status = "CONNU"
        else:
            p = 0.12 + 0.8 * fill_rate.get((t, key), 0.0)
            r = rng.random()
            if r < p:
                value = sample_value(kind_for(t, key), rng, fl, key); status = "CONNU"
            elif r < p + 0.08 * (1 - p):
                value = "–"; status = "NON_FOURNI"
            elif r < p + 0.12 * (1 - p):
                value = str(rng.choice(["?", "inconnu", "Inconnu", "NSP", "unknown", "غير معروف"])); status = "INCONNU"
            else:
                value = None; status = "NON_FOURNI"
        rec = dict(key=key, type="identifier" if ident else "text", value=value, status=status, bbox=None, lang=fl)
        entry = None
        if bank and status == "CONNU" and not ident and bank.get(kind_for(t, key)["kind"]) and rng.random() < bank_p:
            cands = bank[kind_for(t, key)["kind"]]
            entry = cands[int(rng.integers(len(cands)))]
            value, rec["value"], rec["lang"] = entry["text"], entry["text"], entry.get("lang", fl)
        if value:
            fpath = ar_font if is_arabic(value) else main_font
            # rare: font lacking accents (glyph left blank, like NanumPen in the specimen)
            h = base_h * rng.uniform(0.9, 1.1)
            zw, zh = x1 - x0, y1 - y0
            if entry is not None:
                patch = bank_patch(entry, h, ink)
                if patch.width > zw * 1.25 and zw > 40:
                    patch = bank_patch(entry, h * zw * 1.15 / patch.width, ink)
                fpath = entry["path"]
            else:
                patch = render_text(value, fpath, h, ink, rng, cov[fpath], style)
                if patch.width > zw * 1.25 and zw > 40:
                    f = zw * 1.15 / patch.width
                    patch = render_text(value, fpath, h * f, ink, rng, cov[fpath], style)
            tall = zh > 120
            px = x0 + rng.uniform(3, 14)
            if is_arabic(value) and rng.random() < 0.6:
                px = max(x0 + 3, x1 - patch.width - rng.uniform(3, 14))     # right-aligned RTL
            py = (y0 + rng.uniform(10, 40)) if tall else (y0 + y1) / 2 - patch.height / 2 + rng.uniform(-4, 4)
            px, py = int(px), int(py)
            page.paste(patch, (px, py), patch)
            a = np.asarray(patch)[..., 3]
            ys, xs = np.nonzero(a > 40)
            if len(xs):
                rec["bbox"] = [px + int(xs.min()), py + int(ys.min()), px + int(xs.max()), py + int(ys.max())]
            rec["font"] = Path(fpath).name
        fields.append(rec)
    return np.asarray(page), fields, dict(lang=lang, printed=printed, font=Path(main_font).name, ink=ink,
                                          writer=bool(style))


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    out = LOCAL / "synth_preview"
    out.mkdir(exist_ok=True)
    for t in (2, 3, 6):
        for lang in ("fr", "ar"):
            img, f, meta = synth_page(t, rng, lang=lang)
            Image.fromarray(img).save(out / f"p{t}_{lang}.png")
            print(t, lang, meta, sum(1 for x in f if x["value"]))
