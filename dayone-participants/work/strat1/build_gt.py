"""Strategy 1: exact ground truth from the specimen PDF text layer + vector checkboxes.

For every one of the 80 pages (10 patients x 8 page types) we:
  * take printed labels (Helvetica spans) as anchors,
  * group handwriting-font characters into value runs,
  * assign each run to a field key: table cell (row label x column header) or inline label,
  * read checkbox squares and blue tick strokes from the vector drawings,
  * write work/shared/gt/page_NN.json + index.csv + schema.json.

Identifier fields (name, CIN, address, phone, husband's name) keep their key and bbox (for
masking) but their value is never written.
"""
from __future__ import annotations

import collections
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

import pymupdf as fitz

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
from common import GT_DIR, IDENTIFIER_LABELS, PAGE_TYPES, PDF, REG, SCALE, slug  # noqa: E402

DASHES = {"-", "–", "—"}

# ---------------------------------------------------------------- tables per page type
# Each table: header labels (left->right), row labels (top->bottom) or None (single row),
# vertical region delimited by a start label and an end label, optional x-range.
TABLES = {
    2: [
        dict(name="atcd_familiaux", start="famille_de_la_femme", end="antecedents_obstetricaux",
             xmax=300, headers=["famille_de_la_femme", "mari_famille"],
             rows=["hta", "diabete", "maladies_hereditaires", "malformations", "allergie_s"]),
        dict(name="atcd_femme", start="medicaux", end="antecedents_obstetricaux", xmin=300,
             headers=["medicaux", "chirurgicaux", "gynecologiques"], rows=None),
        dict(name="atcd_obstetricaux", start="nature", end="deroulement_des_accouchements_anterieurs",
             headers=["nombre", "date", "lieu", "age_gestationnel_sa"],
             rows=["avortement", "accouchement_premature", "mort_foetale_in_utero", "autres_a_preciser"]),
        dict(name="accouchements_anterieurs", start="accouch_1", end="gestation",
             headers=["accouch_1", "accouch_2", "accouch_3", "accouch_4", "accouch_5"],
             rows=["date", "modalite_d_extraction", "si_cesarienne_indication", "complication_type",
                   "poids_nouveau_ne_s", "compl_nouveau_ne_type"]),
    ],
    3: [
        dict(name="visites", start="prestations_visites", end=None,
             headers=["visite_1", "visite_2", "visite_3", "visite_1", "visite_2", "visite_3",
                      "7eme_mois", "8eme_mois", "9eme_mois"],
             col_keys=["t1_v1", "t1_v2", "t1_v3", "t2_v1", "t2_v2", "t2_v3", "t3_m7", "t3_m8", "t3_m9"],
             rows="auto"),
    ],
}
SECTION_ROWS = {"examen_clinique", "examen_biologique", "traitement", "examen_fait_par_section"}


def spans_of(page):
    out = []
    for b in page.get_text("rawdict")["blocks"]:
        for l in b.get("lines", []):
            for s in l["spans"]:
                out.append(s)
    return out


def label_spans(spans):
    labs = []
    for s in spans:
        if "Helvetica" not in s["font"]:
            continue
        text = "".join(c["c"] for c in s["chars"]).strip()
        if not text:
            continue
        labs.append(dict(text=text, key=slug(text), bbox=tuple(s["bbox"]), bold="Bold" in s["font"]))
    labs.sort(key=lambda d: (round(d["bbox"][1]), d["bbox"][0]))
    return labs


def value_runs(spans):
    """Group handwriting chars into runs (one line, contiguous)."""
    chars = []
    for s in spans:
        if "Helvetica" in s["font"]:
            continue
        for c in s["chars"]:
            chars.append(dict(c=c["c"], bbox=tuple(c["bbox"]), font=s["font"]))
    chars.sort(key=lambda c: ((c["bbox"][1] + c["bbox"][3]) / 2, c["bbox"][0]))
    lines = []
    for c in chars:
        cy = (c["bbox"][1] + c["bbox"][3]) / 2
        for ln in lines:
            if abs(ln["cy"] - cy) < 5:
                ln["chars"].append(c)
                break
        else:
            lines.append(dict(cy=cy, chars=[c]))
    runs = []
    for ln in lines:
        cs = sorted(ln["chars"], key=lambda c: c["bbox"][0])
        cur = [cs[0]]
        for c in cs[1:]:
            if c["bbox"][0] - cur[-1]["bbox"][2] > 4:
                runs.append(cur)
                cur = [c]
            else:
                cur.append(c)
        runs.append(cur)
    out = []
    for r in runs:
        text = "".join(c["c"] for c in r)
        x0 = min(c["bbox"][0] for c in r); y0 = min(c["bbox"][1] for c in r)
        x1 = max(c["bbox"][2] for c in r); y1 = max(c["bbox"][3] for c in r)
        if text.strip():
            out.append(dict(text=text.strip(), bbox=(x0, y0, x1, y1), font=r[0]["font"], chars=r))
    return out


def make_run(cs):
    text = "".join(c["c"] for c in cs)
    return dict(text=text.strip(), font=cs[0]["font"], chars=cs,
                bbox=(min(c["bbox"][0] for c in cs), min(c["bbox"][1] for c in cs),
                      max(c["bbox"][2] for c in cs), max(c["bbox"][3] for c in cs)))


def split_at_columns(runs, hx, y0, y1):
    """Values overflowing into the next cell: split where a char starts exactly at a column start."""
    out = []
    for r in runs:
        if not (y0 < cy(r["bbox"]) < y1):
            out.append(r); continue
        cur = [r["chars"][0]]
        for c in r["chars"][1:]:
            if any(h - 3.5 <= c["bbox"][0] <= h + 3.2 and cur[0]["bbox"][0] < h - 15 for h in hx) and c["c"] != " ":
                out.append(make_run(cur)); cur = [c]
            else:
                cur.append(c)
        out.append(make_run(cur))
    return [r for r in out if r["text"]]


def find_label(labs, key, after_y=-1):
    for l in labs:
        if l["key"].startswith(key) and l["bbox"][1] >= after_y:
            return l
    return None


def cy(b):
    return (b[1] + b[3]) / 2


def assign_tables(ptype, labs, runs):
    """Return {field_key: [runs]} for table cells, plus set of all cell keys and used run ids."""
    cells = collections.defaultdict(list)
    all_keys = {}
    used = set()
    for t in TABLES.get(ptype, []):
        start = find_label(labs, t["start"])
        if start is None:
            raise RuntimeError(f"table start {t['start']} not found")
        y0 = start["bbox"][1] - 2
        end = find_label(labs, t["end"], after_y=y0) if t["end"] else None
        y1 = end["bbox"][1] - 1 if end else 10_000
        xmin, xmax = t.get("xmin", 0), t.get("xmax", 10_000)
        # headers: on the same line as start, in x order, matching names
        hdr_line = [l for l in labs if abs(cy(l["bbox"]) - cy(start["bbox"])) < 6 and xmin <= l["bbox"][0] < xmax]
        hdr_line.sort(key=lambda l: l["bbox"][0])
        heads = []
        hi = 0
        for l in hdr_line:
            if hi < len(t["headers"]) and l["key"].startswith(t["headers"][hi]):
                heads.append(l); hi += 1
        if len(heads) != len(t["headers"]):
            raise RuntimeError(f"headers of {t['name']} found {len(heads)}")
        col_keys = t.get("col_keys") or [h for h in t["headers"]]
        hx = [h["bbox"][0] for h in heads]
        runs[:] = split_at_columns(runs, hx, y0, y1)
        # rows
        if t["rows"] is None:
            rows = [dict(key="contenu", bbox=(0, y0, 0, y1))]
        elif t["rows"] == "auto":
            rows = [l for l in labs if l["bbox"][0] < hx[0] - 20 and l["bbox"][1] > start["bbox"][3]
                    and l["bbox"][1] < y1 and not l["key"].startswith("document_synthetique")]
            rows = [dict(l) for l in rows]
            for r in rows:
                if r["bold"]:
                    r["section"] = True
        else:
            rows = []
            for rk in t["rows"]:
                cand = [l for l in labs if l["key"].startswith(rk) and y0 < l["bbox"][1] < y1
                        and xmin <= l["bbox"][0] < xmax and l["bbox"][0] < hx[0] - 5]
                if not cand:
                    raise RuntimeError(f"row {rk} of {t['name']} not found")
                rows.append(dict(cand[0]))
        real_rows = [r for r in rows if not r.get("section")]
        # dedupe row keys
        seen = collections.Counter()
        for r in real_rows:
            seen[r["key"]] += 1
            if seen[r["key"]] > 1:
                r["key"] = f"{r['key']}_{seen[r['key']]}"
        for r in real_rows:
            for ck in col_keys:
                all_keys[f"{t['name']}.{r['key']}.{ck}"] = "text"
        for i, run in enumerate(runs):
            b = run["bbox"]
            if not (y0 < cy(b) < y1 and xmin <= b[0] < xmax):
                continue
            if b[0] < hx[0] - 12:
                continue
            ci = max(j for j in range(len(hx)) if hx[j] - 12 <= b[0] + 2) if b[0] + 2 >= hx[0] - 12 else 0
            if t["rows"] is None:
                r = real_rows[0]
                if cy(b) < start["bbox"][3]:
                    continue
            else:
                r = min(real_rows, key=lambda r: abs(cy(r["bbox"]) - cy(b)))
                if abs(cy(r["bbox"]) - cy(b)) > 14:
                    continue
            cells[f"{t['name']}.{r['key']}.{col_keys[ci]}"].append(run)
            used.add(i)
    return cells, all_keys, used


def inline_label_key(labs, run, occ_keys):
    b = run["bbox"]
    cands = [l for l in labs if abs(cy(l["bbox"]) - cy(b)) < 8 and l["bbox"][2] <= b[0] + 3]
    if not cands:
        # value written on the line below its label (e.g. "Pourquoi ?")
        above = [l for l in labs if 0 < b[1] - l["bbox"][3] < 25 and l["bbox"][0] <= b[0] + 5
                 and l["text"].rstrip().endswith(("?", ":"))]
        if not above:
            return None
        return occ_keys[id(max(above, key=lambda l: l["bbox"][3]))]
    lab = max(cands, key=lambda l: l["bbox"][2])
    return occ_keys[id(lab)]


def occurrence_keys(labs):
    cnt = collections.Counter()
    out = {}
    for l in labs:
        k = l["key"]
        cnt[k] += 1
        out[id(l)] = k if cnt[k] == 1 else f"{k}_{cnt[k]}"
    return out


def checkboxes(page, labs, occ_keys):
    draws = page.get_drawings()
    boxes, ticks = [], []
    form_col = collections.Counter(tuple(round(v, 2) for v in d["color"]) for d in draws
                                   if d["type"] == "s" and d.get("color")).most_common(1)[0][0]
    for d in draws:
        col = tuple(round(v, 2) for v in (d.get("color") or (0, 0, 0)))
        r = d["rect"]
        if d["type"] == "s" and col != form_col and r.width < 20 and r.height < 20:   # pen stroke = tick
            ticks.append(r)
        elif (d["type"] == "s" and len(d["items"]) == 4 and 6 <= r.width <= 11 and 6 <= r.height <= 11):
            boxes.append(r)
    out = []
    for bx in boxes:
        bcy = (bx.y0 + bx.y1) / 2
        right = [l for l in labs if abs(cy(l["bbox"]) - bcy) < 7 and bx.x1 - 1 <= l["bbox"][0] <= bx.x1 + 25]
        left = [l for l in labs if abs(cy(l["bbox"]) - bcy) < 7 and l["bbox"][2] <= bx.x0 + 1]
        if right:
            lab = min(right, key=lambda l: l["bbox"][0])
        elif left:
            lab = max(left, key=lambda l: l["bbox"][2])
        else:
            lab = None
        # group label: nearest preceding label ending with ':' (same line or above, left of the box)
        grp = None
        for l in labs:
            if l["text"].rstrip().endswith(":") and l["bbox"][1] <= bx.y1 + 2 and l is not lab:
                grp = l
        ticked = any(bx.x0 - 3 <= (t.x0 + t.x1) / 2 <= bx.x1 + 3 and bx.y0 - 3 <= (t.y0 + t.y1) / 2 <= bx.y1 + 3
                     for t in ticks)
        out.append(dict(rect=tuple(bx), label=lab, group=grp, ticked=ticked))
    out.sort(key=lambda d: (round(d["rect"][1] / 4), d["rect"][0]))
    keys = collections.Counter()
    res = []
    for d in out:
        lk = slug(d["label"]["text"]) if d["label"] else "box"
        k = f"cb.{lk}"
        keys[k] += 1
        if keys[k] > 1:
            k = f"{k}_{keys[k]}"
        res.append(dict(key=k, value=d["ticked"], bbox=d["rect"],
                        label=d["label"]["text"] if d["label"] else None,
                        group=d["group"]["text"] if d["group"] else None))
    return res


def px(b):
    return [round(v * SCALE, 1) for v in b]


def union(bs):
    return (min(b[0] for b in bs), min(b[1] for b in bs), max(b[2] for b in bs), max(b[3] for b in bs))


def main():
    doc = fitz.open(PDF)
    GT_DIR.mkdir(parents=True, exist_ok=True)
    pages = []
    unassigned = []
    for n in range(len(doc)):
        page = doc[n]
        ptype = n % 8 + 1
        patient = n // 8 + 1
        spans = spans_of(page)
        labs = label_spans(spans)
        runs = value_runs(spans)
        occ = occurrence_keys(labs)
        cells, table_keys, used = assign_tables(ptype, labs, runs)
        inline = collections.defaultdict(list)
        for i, r in enumerate(runs):
            if i in used:
                continue
            k = inline_label_key(labs, r, occ)
            if k is None:
                unassigned.append((n + 1, r["text"]))
                continue
            inline[k].append(r)
        cbs = checkboxes(page, labs, occ)
        fonts = collections.Counter(r["font"] for r in runs)
        pages.append(dict(n=n + 1, ptype=ptype, patient=patient, cells=cells, table_keys=table_keys,
                          inline=inline, cbs=cbs, font=fonts.most_common(1)[0][0] if fonts else None,
                          labs=labs, occ=occ))
    # schema = union over patients of inline keys + full table grids + checkboxes
    schema = {}
    for p in pages:
        sch = schema.setdefault(p["ptype"], {})
        for k in p["table_keys"]:
            sch[k] = "text"
        for k in p["inline"]:
            sch[f"inline.{k}"] = "text"
        for c in p["cbs"]:
            sch[c["key"]] = "checkbox"
    # inline zones: label bbox per page to support NON_FOURNI zones
    vocab_fix = build_vocab(pages)
    index_rows = []
    hashes = {}
    for f in sorted(REG.glob("dossiers_specimen_10_patientes-*.png")):
        m = re.match(r"dossiers_specimen_10_patientes-(\d+)", f.name)
        hashes.setdefault(int(m.group(1)), []).append((f.name, hashlib.sha256(f.read_bytes()).hexdigest()))
    for p in pages:
        fields = []
        for key, ftype in sorted(schema[p["ptype"]].items()):
            if ftype == "checkbox":
                cb = next((c for c in p["cbs"] if c["key"] == key), None)
                if cb is None:
                    continue
                fields.append(dict(key=key, type="checkbox", value=bool(cb["value"]), status="CONNU",
                                   bbox=px(cb["bbox"]), label=cb["label"], group=cb["group"]))
                continue
            if key.startswith("inline."):
                rs = p["inline"].get(key[7:], [])
                lab = next((l for l in p["labs"] if p["occ"][id(l)] == key[7:]), None)
            else:
                rs = p["cells"].get(key, [])
                lab = None
            rs = sorted(rs, key=lambda r: (round(cy(r["bbox"]) / 6), r["bbox"][0]))
            text = " ".join(r["text"] for r in rs).strip()
            text = fix_text(text, vocab_fix)
            ident = key.startswith("inline.") and key[7:].split("_2")[0] in IDENTIFIER_LABELS
            if not text:
                status, value = "NON_FOURNI", None
            elif text in DASHES:
                status, value = "NON_FOURNI", None
            else:
                status, value = "CONNU", text
            f = dict(key=key, type="text", value=value, status=status,
                     bbox=px(union([r["bbox"] for r in rs])) if rs else None,
                     label_bbox=px(lab["bbox"]) if lab else None,
                     raw="–" if text in DASHES else None)
            if ident:
                f.update(identifier=True, value=None, raw=None)
            fields.append(f)
        out = dict(page=p["n"], patient=p["patient"], page_type=p["ptype"], page_type_name=PAGE_TYPES[p["ptype"]],
                   font=p["font"], png=[h[0] for h in hashes[p["n"]]], fields=fields)
        (GT_DIR / f"page_{p['n']:02d}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        for name, h in hashes[p["n"]]:
            index_rows.append(dict(png=name, page=p["n"], patient=p["patient"], page_type=p["ptype"],
                                   font=p["font"], sha256=h))
    with open(GT_DIR / "index.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(index_rows[0]))
        w.writeheader(); w.writerows(index_rows)
    (GT_DIR / "schema.json").write_text(json.dumps({str(k): v for k, v in schema.items()}, ensure_ascii=False, indent=1), encoding="utf-8")
    print("pages", len(pages), "schema sizes", {k: len(v) for k, v in schema.items()})
    print("unassigned runs", len(unassigned), unassigned[:20])


ACCENTS = "éèêëàâäîïôöùûüçÉÈÊÀÂÎÔÛÇœ"


def build_vocab(pages):
    words = collections.Counter()
    for p in pages:
        for rs in list(p["cells"].values()) + list(p["inline"].values()):
            for r in rs:
                for w in re.findall(r"\S+", r["text"]):
                    if "\x00" not in w:
                        words[w] += 1
    return words


def fix_text(text, vocab):
    if "\x00" not in text:
        return text
    out = []
    for w in text.split(" "):
        if w and set(w) == {"\x00"}:
            out.append("–")          # en-dash glyph missing from the font
            continue
        if "\x00" in w:
            pat = re.compile("^" + re.escape(w).replace("\\\x00", "\x00").replace("\x00", f"[{ACCENTS}]") + "$")
            cands = [v for v in vocab if pat.match(v)]
            if cands:
                w = max(cands, key=lambda v: vocab[v])
            else:
                w = w.replace("\x00", "é")
        out.append(w)
    return " ".join(out)


if __name__ == "__main__":
    main()
