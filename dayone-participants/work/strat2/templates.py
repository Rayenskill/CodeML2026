"""Strategy 2 (part 1): page templates and field zones in template pixel coordinates.

* Reference page for type t = patient 1's page t.
* For every specimen page we fit the affine page->reference from matched printed labels
  (layouts differ only by a global similarity transform), so GT boxes of all 10 patients can be
  expressed in reference coordinates.
* Zones: table cells from header/row geometry; inline fields from the label to the right
  (bounded by the next label / the widest value seen); checkboxes from the vector squares.
* Blank template = reference page with handwriting spans redacted (for registration & synthesis).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pymupdf as fitz

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "shared"))
sys.path.insert(0, str(HERE.parents[0] / "strat1"))
import build_gt as B  # noqa: E402
from common import GT_DIR, LOCAL, PDF, SCALE, SHARED  # noqa: E402

TPL_DIR = LOCAL / "templates"


def match_affine(labs, ref):
    """Affine (2x3) mapping page points -> reference points, from labels with identical text."""
    src, dst = [], []
    used = set()
    for l in labs:
        best, bd = None, 1e9
        for j, r in enumerate(ref):
            if r["text"] != l["text"] or j in used:
                continue
            dd = abs(r["bbox"][0] - l["bbox"][0]) + abs(r["bbox"][1] - l["bbox"][1])
            if dd < bd:
                best, bd = j, dd
        if best is not None and bd < 60:
            used.add(best)
            r = ref[best]
            for (x, y), (u, v) in [((l["bbox"][0], l["bbox"][1]), (r["bbox"][0], r["bbox"][1])),
                                   ((l["bbox"][2], l["bbox"][3]), (r["bbox"][2], r["bbox"][3]))]:
                src.append([x, y, 1]); dst.append([u, v])
    A, Y = np.array(src), np.array(dst)
    M, *_ = np.linalg.lstsq(A, Y, rcond=None)
    res = np.abs(A @ M - Y).max()
    return M.T, float(res)          # 2x3


def apply(M, b):
    pts = np.array([[b[0], b[1], 1], [b[2], b[1], 1], [b[0], b[3], 1], [b[2], b[3], 1]]) @ M.T
    return [pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()]


def render_blank(doc_path, pno, out_png):
    doc = fitz.open(doc_path)
    page = doc[pno]
    for s in B.spans_of(page):
        if "Helvetica" in s["font"]:
            continue
        for c in s["chars"]:
            page.add_redact_annot(fitz.Rect(c["bbox"]))
    page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE, graphics=fitz.PDF_REDACT_LINE_ART_NONE)
    # remove pen ticks by painting over them is not possible without touching boxes: keep the
    # vector drawing list and re-draw the page without pen strokes instead.
    pix = page.get_pixmap(dpi=200)
    pix.save(out_png)


def render_blank_clean(pno, out_png):
    """Blank form: printed text + form strokes only (no handwriting, no ticks)."""
    src = fitz.open(PDF)
    page = src[pno]
    out = fitz.open()
    np_ = out.new_page(width=page.rect.width, height=page.rect.height)
    draws = page.get_drawings()
    import collections
    form_col = collections.Counter(tuple(round(v, 2) for v in d["color"]) for d in draws
                                   if d["type"] == "s" and d.get("color")).most_common(1)[0][0]
    shape = np_.new_shape()
    for d in draws:
        col = tuple(round(v, 2) for v in (d.get("color") or (0, 0, 0)))
        if d["type"] == "s" and col != form_col:
            continue
        for it in d["items"]:
            if it[0] == "l":
                shape.draw_line(it[1], it[2])
            elif it[0] == "re":
                shape.draw_rect(it[1])
            elif it[0] == "qu":
                shape.draw_quad(it[1])
            elif it[0] == "c":
                shape.draw_bezier(it[1], it[2], it[3], it[4])
        shape.finish(color=d.get("color"), fill=d.get("fill"), width=d.get("width") or 1,
                     closePath=d.get("closePath", False))
    shape.commit()
    for s in B.spans_of(page):
        if "Helvetica" not in s["font"]:
            continue
        text = "".join(c["c"] for c in s["chars"])
        fn = "hebo" if "Bold" in s["font"] else "helv"
        np_.insert_text(s["chars"][0]["origin"], text, fontname=fn, fontsize=s["size"],
                        color=tuple(((s["color"] >> 16) & 255, (s["color"] >> 8) & 255, s["color"] & 255)) and
                        (((s["color"] >> 16) & 255) / 255, ((s["color"] >> 8) & 255) / 255, (s["color"] & 255) / 255))
    pix = np_.get_pixmap(dpi=200)
    pix.save(out_png)


def main():
    TPL_DIR.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(PDF)
    refs = {t: B.label_spans(B.spans_of(doc[t - 1])) for t in range(1, 9)}
    affines = {}
    for n in range(80):
        t = n % 8 + 1
        M, res = match_affine(B.label_spans(B.spans_of(doc[n])), refs[t])
        affines[n + 1] = dict(M=M.tolist(), res=res)
    print("max affine residual (pt):", max(a["res"] for a in affines.values()))

    templates = {}
    for t in range(1, 9):
        page = doc[t - 1]
        spans = B.spans_of(page)
        labs = refs[t]
        runs = B.value_runs(spans)
        occ = B.occurrence_keys(labs)
        zones = {}
        # --- table cells
        for tb in B.TABLES.get(t, []):
            start = B.find_label(labs, tb["start"])
            y0 = start["bbox"][1] - 2
            end = B.find_label(labs, tb["end"], after_y=y0) if tb["end"] else None
            xmin, xmax = tb.get("xmin", 0), tb.get("xmax", 10_000)
            hdr_line = sorted([l for l in labs if abs(B.cy(l["bbox"]) - B.cy(start["bbox"])) < 6
                               and xmin <= l["bbox"][0] < xmax], key=lambda l: l["bbox"][0])
            heads, hi = [], 0
            for l in hdr_line:
                if hi < len(tb["headers"]) and l["key"].startswith(tb["headers"][hi]):
                    heads.append(l); hi += 1
            hx = [h["bbox"][0] for h in heads]
            col_keys = tb.get("col_keys") or tb["headers"]
            y1 = end["bbox"][1] - 4 if end else 800
            vx = [(it[1].x + it[2].x) / 2 for d in page.get_drawings() for it in d["items"] if it[0] == "l"
                  and abs(it[1].x - it[2].x) < 2.5 and abs(it[1].y - it[2].y) > 15
                  and min(it[1].y, it[2].y) < y1 and max(it[1].y, it[2].y) > y0 and xmin <= it[1].x <= xmax]
            right = max(v for v in vx if v > hx[-1])
            xb = [h - 4 for h in hx] + [right]
            if tb["rows"] is None:
                rows = [dict(key="contenu", bbox=(0, start["bbox"][3] + 4, 0, y1))]
            elif tb["rows"] == "auto":
                rows = [dict(l) for l in labs if l["bbox"][0] < hx[0] - 20 and l["bbox"][1] > start["bbox"][3]
                        and not l["key"].startswith("document_synthetique") and not l["bold"]]
            else:
                rows = [dict(next(l for l in labs if l["key"].startswith(rk) and y0 < l["bbox"][1]
                                  and xmin <= l["bbox"][0] < xmax and l["bbox"][0] < hx[0] - 5)) for rk in tb["rows"]]
            import collections
            seen = collections.Counter()
            for r in rows:
                seen[r["key"]] += 1
                if seen[r["key"]] > 1:
                    r["key"] = f"{r['key']}_{seen[r['key']]}"
            pitch = np.median(np.diff([B.cy(r["bbox"]) for r in rows])) if len(rows) > 1 else 20
            for r in rows:
                if tb["rows"] is None:
                    ry0, ry1 = r["bbox"][1], r["bbox"][3]
                else:
                    ry0, ry1 = B.cy(r["bbox"]) - pitch / 2, B.cy(r["bbox"]) + pitch / 2
                for ci, ck in enumerate(col_keys):
                    zones[f"{tb['name']}.{r['key']}.{ck}"] = dict(type="text", zone=[xb[ci], ry0, xb[ci + 1], ry1])
        # --- inline fields: label -> right
        gt_keys = json.loads((GT_DIR / "schema.json").read_text(encoding="utf-8"))[str(t)]
        for key, typ in gt_keys.items():
            if not key.startswith("inline."):
                continue
            lk = key[7:]
            lab = next(l for l in labs if occ[id(l)] == lk)
            lb = lab["bbox"]
            # max extent of values over the 10 patients (in reference coords)
            x_end, y_lo, y_hi = lb[2] + 60, lb[1] - 5, lb[3] + 5
            below = lb[3] + 25 if lab["text"].rstrip().endswith("?") else None
            for p in range(10):
                g = json.loads((GT_DIR / f"page_{p * 8 + t:02d}.json").read_text(encoding="utf-8"))
                f = next((f for f in g["fields"] if f["key"] == key), None)
                if f and f.get("bbox"):
                    M = np.array(affines[p * 8 + t]["M"])
                    bb = apply(M, [v / SCALE for v in f["bbox"]])
                    x_end = max(x_end, bb[2] + 15)
                    y_lo, y_hi = min(y_lo, bb[1] - 3), max(y_hi, bb[3] + 3)
            ul = [it[2].x for d in page.get_drawings() for it in d["items"] if it[0] == "l"
                  and abs(it[1].y - it[2].y) < 2.5 and lb[3] - 3 <= it[1].y <= lb[3] + 9
                  and lb[2] - 6 <= it[1].x <= lb[2] + 15]
            if ul:
                x_end = max(x_end, max(ul) + 5)
            c = B.cy(lb)
            y_lo, y_hi = max(y_lo, c - 12), min(y_hi, c + 11)
            nxt = [l["bbox"][0] for l in labs if abs(B.cy(l["bbox"]) - B.cy(lb)) < 8 and l["bbox"][0] > lb[2] + 5]
            if nxt:
                x_end = min(x_end, min(nxt) - 2)
            if below:
                zones[key] = dict(type="text", zone=[lb[0], lb[3] + 1, 560, below])
            else:
                zones[key] = dict(type="text", zone=[lb[2] + 1, y_lo, min(x_end, 575), y_hi])
        # --- checkboxes
        for cb in B.checkboxes(page, labs, occ):
            zones[cb["key"]] = dict(type="checkbox", zone=list(cb["bbox"]), label=cb["label"], group=cb["group"])
        # px
        for z in zones.values():
            z["zone_px"] = [round(v * SCALE, 1) for v in z["zone"]]
        missing = set(gt_keys) - set(zones)
        if missing:
            print("type", t, "zones missing for", missing)
        templates[t] = dict(page_type=t, ref_page=t, zones=zones,
                            labels=[dict(text=l["text"], bbox_px=[round(v * SCALE, 1) for v in l["bbox"]]) for l in labs])
        render_blank_clean(t - 1, str(TPL_DIR / f"blank_p{t}.png"))
        doc[t - 1].get_pixmap(dpi=200).save(str(TPL_DIR / f"ref_p{t}.png"))
    (SHARED / "templates.json").write_text(json.dumps({str(k): v for k, v in templates.items()}, ensure_ascii=False),
                                           encoding="utf-8")
    (SHARED / "page_affines.json").write_text(json.dumps(affines), encoding="utf-8")
    print({t: len(v["zones"]) for t, v in templates.items()})


if __name__ == "__main__":
    main()
