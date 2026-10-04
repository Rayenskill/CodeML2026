"""Strategy 2 (part 4): end-to-end zonal extractor.

photo -> register (page type + homography) -> mask identifier zones -> crop every zone
-> CRNN log-probs -> constrained rescoring (strategy 7/11) -> value, status, confidence
-> checkboxes via the OMR model (strategy 18) or ink density.

python extract_zonal.py --model crnn_v1.pt --pages 1-80 --sev 0 --out preds/
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

W = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(W / "shared"), str(W / "strat2"), str(W / "strat4"), str(W / "strat5"), str(W / "strat7"),
                str(W / "strat11"), str(W / "strat18")]
from common import IDENTIFIER_LABELS  # noqa: E402
from crops import TEMPLATES, checkbox_crop, crop_zone, ink_score, new_ink, zone_box  # noqa: E402
from fieldlogic import EMPTY_TAU, decide, decide_multi, kind, status_for, valid_value  # noqa: E402
from register import register, register_any_orientation  # noqa: E402


def is_identifier(key: str) -> bool:
    return key.startswith("inline.") and key[7:].split("_2")[0] in IDENTIFIER_LABELS


def mask_identifiers(warped, t, extra=0):
    """Strategy 10: blank identifier zones before any recognition / storage. `extra` widens the masks when the
    registration is weak (the zones may be off by a few tens of pixels)."""
    out = warped.copy()
    for key, z in TEMPLATES[str(t)]["zones"].items():
        if z["type"] == "text" and is_identifier(key):
            x0, y0, x1, y1 = zone_box(t, key, (2 + extra, 4 + extra, 30 + extra, 4 + extra))
            out[y0:y1, x0:x1] = np.median(out[y0:y1, x0:x1].reshape(-1, 3), 0)
    return out


class Extractor:
    def __init__(self, rec, omr=None, calibrator=None, delta=6.0, tta=False, views=(1.0, 1.2, 1.4)):
        self.rec, self.omr, self.cal, self.delta, self.tta = rec, omr, calibrator, delta, tta
        self.views = tuple(views)

    def checkbox_value(self, warped, t, key):
        crop, box = checkbox_crop(warped, t, key)
        if self.omr is not None:
            p = self.omr.predict(crop)
            return p >= 0.5, float(max(p, 1 - p))
        ink = new_ink(warped, t, box)
        frac = float((ink > 0.3).mean())
        p = 1 / (1 + np.exp(-(frac - 0.06) * 60))
        return p >= 0.5, float(max(p, 1 - p))

    def extract(self, photo, page_type=None, registered=None):
        rotation = None
        if registered is not None:
            reg = registered
        else:                                       # upright first; sideways / upside-down photos as a fallback
            reg, photo, rotation = register_any_orientation(photo, page_type, accept=REG_WEAK, good=REG_WEAK)
        t = reg["page_type"]
        warped = mask_identifiers(reg["warped"], t, extra=0 if reg["score"] >= REG_WEAK else 30)
        self.last_warped = warped                    # identifier-masked, template frame (evidence crops)
        zones = TEMPLATES[str(t)]["zones"]
        keys, crops = [], []
        for key, z in zones.items():
            if z["type"] != "text" or is_identifier(key):
                continue
            c, _ = crop_zone(warped, t, key)
            keys.append(key); crops.append(c)
        # horizontal-stretch views: thin doubled glyphs ("11", "ll") get enough CTC frames in the wider views
        view_lps = []
        for s in self.views:
            cs = crops if s == 1.0 else [cv2.resize(c, (max(8, int(c.shape[1] * s)), c.shape[0])) for c in crops]
            view_lps.append(self.rec.logprobs_tta(cs) if self.tta else self.rec.logprobs(cs))
        lps = [list(v) for v in zip(*view_lps)]
        self.last_lp = dict(zip(keys, lps))          # kept in memory for the consistency pass (strategy 7/13)
        fields = []
        for key, lpv in zip(keys, lps):
            lp = lpv[0]
            _, probs, seqconf = self.rec.decode(lp)
            if len(lpv) > 1:
                value, info = decide_multi(self.rec, lpv, t, key, self.delta)
            else:
                value, info = decide(self.rec, lp, self.rec.decode(lp)[0].strip(), t, key, self.delta)
            greedy = info["greedy"]
            value = value.strip()
            ink = ink_score(warped, t, key)
            feats = dict(seqconf=seqconf, minp=min(probs) if probs else 1.0, n=len(value), snapped=info["snapped"],
                         agree=float(info.get("views_agree", True)),
                         margin=min(info["margin"], 20.0), ink=ink, reg=reg["score"],
                         gap=info["best_score"] - info["empty_score"] if value else info["empty_score"])
            if self.cal:
                conf = self.cal.predict(feats, kind(t, key)["kind"])
            else:                                   # uncalibrated fallback: raw CTC path confidence
                conf = seqconf if value else float(np.exp(info["empty_score"]))
            v = None if value in ("", "–", "-") else value
            st = status_for(v, conf)
            if value == "" and conf < EMPTY_TAU:
                st = "À_RÉVISER"            # read as blank but unsure: a missed faint value is worse than a question
            elif st == "CONNU" and not valid_value(v, kind(t, key)):
                st = "À_RÉVISER"            # breaks its own format / plausible range: never auto-accepted
            fields.append(dict(key=key, type="text", value=v, raw=greedy, status=st, confidence=float(conf),
                               feats=feats))
        for key, z in zones.items():
            if z["type"] != "checkbox":
                continue
            val, conf = self.checkbox_value(warped, t, key)
            fields.append(dict(key=key, type="checkbox", value=bool(val), status="CONNU" if conf > 0.8 else "À_RÉVISER",
                               confidence=conf))
        page = dict(page_type=int(t), registration=dict(score=reg["score"], H=np.asarray(reg["H"]).tolist(),
                                                        rotation=rotation), fields=fields)
        from validator import apply_form_logic, apply_group_logic
        apply_form_logic(page)                      # blank + excluded by the form's logic -> NON_APPLICABLE
        apply_group_logic(page)                     # exclusive checkbox groups: none ticked / several ticked
        page_verdict(page, reg["score"])
        return page


import re as _re
# fields used by the consistency rules (strategy 7/13): only these keep their log-probs across pages
RULE_KEYS = _re.compile(r"inline\.(ddr|date_|age|perimetre_cranien|vu_par)|visites\.(venue_le|age_probable|rendez_vous|hu_cm)\.")


def twin_type_from_content(pred: dict, delivery_date=None):
    """Strategy 13/17: précoce (5/6) vs tardif (7/8) pages share the layout; decide from what is written.
    Newborn page: age in days <= 20 -> précoce. Mother page: consultation - delivery <= 20 days -> précoce."""
    import re
    from validator import pdate
    t = pred["page_type"]
    if t not in (5, 6, 7, 8):
        return None
    # the free reading, not the value constrained by the *assumed* type's grammar: on a page wrongly typed
    # "tardif", "7 jours" is implausible and could be rescored to "71 jours", confirming the wrong type
    raw = {f["key"]: f.get("raw") or f.get("value") for f in pred["fields"]}
    val = {f["key"]: f.get("value") for f in pred["fields"]}
    if t in (6, 8):
        m = re.match(r"\s*(\d+)", raw.get("inline.age") or "")
        if m:
            return 6 if int(m.group(1)) <= 20 else 8
    key = "inline.date_de_la_consultation"                  # a date's grammar does not depend on the page type
    cd = pdate(raw.get(key)) or pdate(val.get(key))
    if cd and delivery_date:
        gap = (cd - delivery_date).days
        if 0 <= gap <= 120:
            early = gap <= 20
            return (5 if early else 7) if t in (5, 7) else (6 if early else 8)
    return None


REG_REJECT, REG_WEAK = 0.45, 0.70      # chosen on the sev 0/2/4 runs: below 0.45 pages read ~0% correctly


def page_verdict(page: dict, reg_score: float):
    """The agent never hides doubt at page level: an unrecognised or badly registered page cannot produce
    confident fields (e.g. a real booklet with another layout, a photo of the wrong page, heavy blur)."""
    if reg_score < REG_REJECT:
        page["page_status"] = "PAGE_NON_RECONNUE"
        for f in page["fields"]:
            f["confidence"] = min(f.get("confidence", 0), 0.2)
            f["status"] = "ILLISIBLE" if f.get("value") not in (None, False) else "À_RÉVISER"
    elif reg_score < REG_WEAK:
        page["page_status"] = "QUALITÉ_FAIBLE"
        for f in page["fields"]:
            f["confidence"] = min(f.get("confidence", 0), 0.6)
            if f.get("status") == "CONNU":
                f["status"] = "À_RÉVISER"
    else:
        page["page_status"] = "OK"
    return page


def run_specimen(ext, pages, sev, seed=0, use_true_type=False, loo_vocab=True):
    """Extract the specimen pages (optionally degraded) -> {page: pred}."""
    from common import GT_DIR, REG
    from degrade import degrade_page
    rng = np.random.default_rng(seed)
    preds = {}
    delivery = {}
    for n in pages:
        g = json.loads((GT_DIR / f"page_{n:02d}.json").read_text(encoding="utf-8"))
        img = cv2.cvtColor(cv2.imread(str(REG / g["png"][0])), cv2.COLOR_BGR2RGB)
        photo, _, _ = degrade_page(img, sev, rng) if sev > 0 else (img, None, None)
        import fieldlogic
        fieldlogic.EXCLUDE_PATIENT = g["patient"] if loo_vocab else None
        preds[n] = ext.extract(photo, g["page_type"] if use_true_type else None)
        if not use_true_type:
            from validator import pdate
            pt = preds[n]["page_type"]
            if pt == 4:
                ctx = {f["key"]: f.get("value") for f in preds[n]["fields"]}
                delivery[g["patient"]] = pdate(ctx.get("inline.date_de_l_accouchement"))
            tt = twin_type_from_content(preds[n], delivery.get(g["patient"]))
            if tt and tt != pt:
                preds[n] = ext.extract(photo, tt)
                preds[n]["twin_fix"] = f"{pt}->{tt}"
        preds[n]["_lp"] = {k: [x.half() for x in v] for k, v in ext.last_lp.items() if RULE_KEYS.search(k)}
        preds[n]["patient"] = g["patient"]
    return preds


def apply_consistency(preds: dict, rec):
    """Strategy 7/13: booklet-level rule checking & likelihood-checked repair, grouped by patient."""
    import collections
    from validator import validate_booklet
    groups = collections.defaultdict(list)
    for n, p in preds.items():
        groups[p.get("patient")].append(p)
    issues = []
    for pages in groups.values():
        issues += validate_booklet(pages, rec)
    return issues
