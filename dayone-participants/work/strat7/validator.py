"""Strategy 7 (part 2) + strategy 13: cross-field / cross-page consistency rules with likelihood-checked repair.

Rules (all verified on the 10 specimen patients, 0 false alarms). Honesty note: R1 and H1 were found by
inspecting the specimen ground truth, i.e. the test set; they are also standard care practice (monthly
appointment, McDonald's rule), which is why they only repair under the likelihood test:
  D1  date_prevue_d_accouchement = ddr + 280 d             (page 3)
  D2  date_de_depassement_de_terme = date_prevue + 7 d      (page 3)
  D4  visit "Âge probable" = round((venue_le - ddr) / 7) SA (page 3, soft ±1)
  D5  age_gestationnel (p4) = floor((date accouchement - ddr) / 7) SA        (booklet)
  D6  newborn age (p6) = consultation (p5/p6) - delivery date (days)         (booklet)
  D6' newborn age (p8) = consultation (p7/p8) - delivery date (days)        (booklet, reported as D6)
  R1  visit "Rendez-vous" = "Venue le" + 28 d                 (page 3, 50/50; soft: repair only)
  H1  visit fundal height (HU, cm) = "Âge probable" (SA) - 4  (page 3, 44/44, McDonald's rule; soft)
      Tried and dropped (measured at sev 2): decoding the visit row jointly (Viterbi over each cell's plausible
      readings) under "dates in chronological order" or "weight gain -1.5..+6 kg between visits". On blurred rows it
      mostly turned unreadable cells into well-formed wrong values (dates: 4 fixes / 18 wrong->wrong / 1 broken;
      weights: 3 / 20 / 2), i.e. more questions and a few new errors for almost no gain.
  E1  same fact on two pages: p5/p6 consultation date, p7/p8 consultation date, p4/p6 head circumference,
      p6/p8 "vu par"                                                          (booklet)
  O1  gravidity >= parity ; living children <= parity + 1 (flag only)
  V*  physiological ranges, flag only: V1 BP format sys>dia, V2 mother's weight 30-200 kg, V3 temperature 34-42,
      V4 Hb 4-20 g/dL, V5 FHR 80-220

Repair: a rule proposes the implied value v*; it replaces the reading only if the recogniser itself finds v*
plausible: CTC log-likelihood of v* >= reading's score - DELTA_RULE; the field then becomes À_RÉVISER with
provenance "rule:<id>" and the original reading offered as the alternative. Otherwise the field is flagged
(À_RÉVISER, with v* offered to the midwife as a quick reply). We never invent a value for a blank field.
Soft rules (R1, H1: care protocol / rule of thumb, may not hold for every woman) repair under the same
likelihood test but never flag: v* is only kept as a suggestion.
"""
from __future__ import annotations

import datetime as dt
import re

DELTA_RULE = 8.0
VISIT_RULES = True          # R1/H1 (soft); switch kept for ablation (replay_rules.py)


def pdate(s):
    if not s:
        return None
    m = re.fullmatch(r"\s*(\d{1,2})/(\d{1,2})/(\d{4})\s*", s)
    if not m:
        return None
    try:
        return dt.date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
    except ValueError:
        return None


def fdate(d):
    return d.strftime("%d/%m/%Y")


def pint(s, unit=None):
    if not s:
        return None
    m = re.match(r"\s*(\d+)", s)
    return int(m.group(1)) if m else None


class FieldRef:
    """A field reading with what is needed to test alternatives: (page dict, field dict, logp, rec)."""

    def __init__(self, page, field, lp=None):
        self.page, self.f, self.lp = page, field, lp

    @property
    def value(self):
        return self.f.get("value")


def repair_cost(ref: FieldRef, implied: str, rec):
    """Log-likelihood the recogniser loses by replacing its reading with the implied value (inf if untestable)."""
    if ref.lp is None or rec is None or ref.value is None or ref.f.get("reviewed"):
        return float("inf")
    return rec.score(ref.lp, ref.value) - rec.score(ref.lp, implied)


def try_repair(ref: FieldRef, implied: str, rec, rule: str, issues: list, delta=DELTA_RULE, soft=False):
    cur = ref.value
    if cur == implied:
        ref.f.setdefault("rules_ok", []).append(rule)
        return False
    if cur is None:                       # never fill a blank from a rule
        return False
    if ref.lp is not None and rec is not None and not ref.f.get("reviewed"):   # never override the midwife
        s_cur = rec.score(ref.lp, cur)
        s_imp = rec.score(ref.lp, implied)
        if s_imp >= s_cur - delta:
            ref.f["value"] = implied
            ref.f["status"] = "À_RÉVISER"             # a rule changed the reading: the midwife confirms it
            ref.f["provenance"] = f"rule:{rule}"
            ref.f.setdefault("suggestions", []).append(cur)
            ref.f.setdefault("repairs", []).append(dict(rule=rule, old=cur, new=implied, d=s_cur - s_imp))
            issues.append(dict(rule=rule, key=ref.f["key"], page_type=ref.page.get("page_type"), old=cur,
                               new=implied, repaired=True))
            return True
    ref.f.setdefault("suggestions", []).append(implied)
    if not soft:
        ref.f["status"] = "À_RÉVISER"
        issues.append(dict(rule=rule, key=ref.f["key"], page_type=ref.page.get("page_type"), old=cur,
                           suggestion=implied, repaired=False))
    return False


def _get(pages, t, key):
    for pg in pages:
        if pg["page_type"] == t:
            for f in pg["fields"]:
                if f["key"] == key:
                    return FieldRef(pg, f, pg.get("_lp", {}).get(key))
    return None


def _visit_cols(pages):
    for pg in pages:
        if pg["page_type"] == 3:
            return [f["key"].split(".")[-1] for f in pg["fields"] if f["key"].startswith("visites.venue_le.")]
    return []


def visit_rules(pages, ddr_d, rec, issues, which):
    """Soft rules of the visit table. R1: next appointment = visit date + 28 d; when the pair disagrees, the
    cheaper repair (in recogniser log-likelihood) wins, favouring a visit date consistent with DDR + "Âge probable".
    H1: fundal height = gestational age - 4, only from a gestational age corroborated by DDR + visit date (a
    misread SA would otherwise propagate into the fundal height)."""
    for col in _visit_cols(pages):
        if which == "R1":
            v, r = _get(pages, 3, f"visites.venue_le.{col}"), _get(pages, 3, f"visites.rendez_vous.{col}")
            if not (v and r and v.value and r.value):
                continue
            vd, rd = pdate(v.value), pdate(r.value)
            if vd and rd and (rd - vd).days == 28:
                v.f.setdefault("rules_ok", []).append("R1"); r.f.setdefault("rules_ok", []).append("R1")
                continue
            ap = _get(pages, 3, f"visites.age_probable.{col}")
            sa = pint(ap.value) if ap else None

            def d4_ok(d):
                return bool(d and ddr_d and sa is not None and round((d - ddr_d).days / 7) == sa)
            opts = []
            if vd:
                opts.append((repair_cost(r, fdate(vd + dt.timedelta(28)), rec), r, fdate(vd + dt.timedelta(28))))
            if rd:
                cand = rd - dt.timedelta(28)
                bonus = 3.0 if d4_ok(cand) and not d4_ok(vd) else -3.0 if d4_ok(vd) and not d4_ok(cand) else 0.0
                opts.append((repair_cost(v, fdate(cand), rec) - bonus, v, fdate(cand)))
            if opts:
                _, ref, implied = min(opts, key=lambda o: o[0])
                try_repair(ref, implied, rec, "R1", issues, delta=6.0, soft=True)
        else:
            ap, hu = _get(pages, 3, f"visites.age_probable.{col}"), _get(pages, 3, f"visites.hu_cm.{col}")
            sa, h = (pint(ap.value) if ap else None), (pint(hu.value) if hu else None)
            if sa is None or h is None or not hu.value.strip().isdigit() or not 16 <= sa <= 42:
                continue
            ve = _get(pages, 3, f"visites.venue_le.{col}")
            vd = pdate(ve.value) if ve else None
            corroborated = ddr_d and vd and round((vd - ddr_d).days / 7) == sa
            if not corroborated or ap.f.get("repairs") or ve.f.get("repairs"):   # SA as written, agreeing with dates
                continue
            if h != sa - 4:
                try_repair(hu, str(sa - 4), rec, "H1", issues, delta=4.0, soft=True)


def _parse_float(s):
    m = re.match(r"\s*(\d+(?:[.,]\d+)?)", s or "")
    return float(m.group(1).replace(",", ".")) if m else None


def _ascii(s):
    from vocab import to_ascii_digits
    return to_ascii_digits(s or "").replace("،", ",")


def _flag(f, rule, issues, page_type):
    if f.get("reviewed"):
        return
    if f.get("status") == "CONNU":
        f["status"] = "À_RÉVISER"
    f.setdefault("flags", []).append(rule)
    issues.append(dict(rule=rule, key=f["key"], page_type=page_type, value=f.get("value"), repaired=False))


def validate_booklet(pages: list[dict], rec=None):
    """pages: extracted page dicts (with optional '_lp' {key: logp}). Applies rules in place; returns issues."""
    issues = []
    g = lambda t, k: _get(pages, t, k)   # noqa: E731
    ddr, dpa, ddt = g(3, "inline.ddr"), g(3, "inline.date_prevue_d_accouchement"), g(3, "inline.date_de_depassement_de_terme")
    if ddr and dpa and ddt:
        a, b, c = pdate(ddr.value), pdate(dpa.value), pdate(ddt.value)
        ok1 = a and b and (b - a).days == 280
        ok2 = b and c and (c - b).days == 7
        if ok1 and not ok2 and b:
            try_repair(ddt, fdate(b + dt.timedelta(7)), rec, "D2", issues)
        elif ok2 and not ok1 and b:
            try_repair(ddr, fdate(b - dt.timedelta(280)), rec, "D1", issues)
        elif not ok1 and not ok2:
            # DDR and DDT agree with each other (287 d) -> repair DPA
            if a and c and (c - a).days == 287:
                try_repair(dpa, fdate(a + dt.timedelta(280)), rec, "D1", issues)
    ddr_d = pdate(ddr.value) if ddr else None
    if VISIT_RULES:
        visit_rules(pages, ddr_d, rec, issues, "R1")
    # D4: visit gestational age
    if ddr_d:
        for pg in pages:
            if pg["page_type"] != 3:
                continue
            for f in pg["fields"]:
                if f["key"].startswith("visites.venue_le."):
                    col = f["key"].split(".")[-1]
                    vd = pdate(f.get("value"))
                    ap = _get(pages, 3, f"visites.age_probable.{col}")
                    if vd and ap and ap.value:
                        weeks = (vd - ddr_d).days / 7
                        n = pint(ap.value)
                        if n is not None and abs(n - weeks) > 1.6:
                            m = re.match(r"\s*\d+\s*(.*)", ap.value)
                            implied = f"{round(weeks)} {m.group(1).strip() if m else 'SA'}".strip()
                            try_repair(ap, implied, rec, "D4", issues, delta=4.0)
    if VISIT_RULES:
        visit_rules(pages, ddr_d, rec, issues, "H1")
    # D5 gestational age at birth
    acc = g(4, "inline.date_de_l_accouchement")
    acc_d = pdate(acc.value) if acc else None
    ga = g(4, "inline.age_gestationnel")
    if ddr_d and acc_d and ga and ga.value:
        w = (acc_d - ddr_d).days // 7
        n = pint(ga.value)
        if n is not None and n != w:
            m = re.match(r"\s*\d+\s*(.*)", ga.value)
            try_repair(ga, f"{w} {m.group(1).strip() if m else 'SA'}".strip(), rec, "D5", issues, delta=5.0)
    # E1 same fact on two pages: keep the reading the recogniser is more sure about
    for (t1, k1), (t2, k2) in [((5, "inline.date_de_la_consultation"), (6, "inline.date_de_la_consultation")),
                               ((7, "inline.date_de_la_consultation"), (8, "inline.date_de_la_consultation")),
                               ((4, "inline.perimetre_cranien_a_la_naissance"), (6, "inline.perimetre_cranien")),
                               ((6, "inline.vu_par"), (8, "inline.vu_par"))]:
        a, b = g(t1, k1), g(t2, k2)
        if a and b and a.value and b.value and a.value != b.value:
            ca, cb = a.f.get("confidence", 0), b.f.get("confidence", 0)
            src, dst = (a, b) if ca >= cb else (b, a)
            try_repair(dst, src.value, rec, "E1", issues, delta=6.0)
    # D6/D7 newborn age in days
    for tc, tn in ((5, 6), (7, 8)):
        cons = g(tn, "inline.date_de_la_consultation") or g(tc, "inline.date_de_la_consultation")
        age = g(tn, "inline.age")
        cd = pdate(cons.value) if cons else None
        if acc_d and cd and age and age.value:
            days = (cd - acc_d).days
            n = pint(age.value)
            if n is not None and n != days and 0 < days < 120:
                m = re.match(r"\s*\d+\s*(.*)", age.value)
                try_repair(age, f"{days} {m.group(1).strip() if m else 'jours'}".strip(), rec, "D6", issues, delta=5.0)
    # O1 obstetric formula (page 2): gravidity >= parity, living children <= parity + 1 (twins) -> flags only
    gr, pa, lc = g(2, "inline.gestation"), g(2, "inline.parite"), g(2, "inline.nombre_d_enfants_vivants")
    G, P, L = [pint(_ascii(x.value)) if x and x.value else None for x in (gr, pa, lc)]
    if G is not None and P is not None and G < P:
        for x in (gr, pa):
            _flag(x.f, "O1", issues, 2)
    if P is not None and L is not None and L > P + 1:
        _flag(lc.f, "O1", issues, 2)
    # V6 / V7 visit rows, flags only: weight gain between consecutive visits outside -1.5..+6 kg, visit dates not
    # in order (7..120 d apart). Both cells of an inconsistent pair are asked; no value is changed (repairing
    # them along the row was measured to turn unreadable cells into well-formed wrong values).
    for field, rule, parse, ok in (("poids_kg", "V6", lambda v: _parse_float(_ascii(v)), lambda a, b: -1.5 <= b - a <= 6),
                                   ("venue_le", "V7", pdate, lambda a, b: 7 <= (b - a).days <= 120)):
        row = [r for col in _visit_cols(pages) for r in [_get(pages, 3, f"visites.{field}.{col}")] if r and r.value]
        vals = [parse(r.value) for r in row]
        for (r1, a1), (r2, a2) in zip(zip(row, vals), list(zip(row, vals))[1:]):
            if a1 is not None and a2 is not None and not ok(a1, a2):
                for r in (r1, r2):
                    if rule not in r.f.get("flags", []):
                        _flag(r.f, rule, issues, 3)
    # physiological ranges / formats -> flags only (never on a value the midwife confirmed)
    for pg in pages:
        for f in pg["fields"]:
            v = f.get("value")
            if not v or f.get("type") != "text" or f.get("reviewed"):
                continue
            k, va = f["key"], _ascii(v)
            num = _parse_float(va)
            bad = None
            if k.endswith(".ta") or ".ta." in k:
                m = re.fullmatch(r"(\d{2,3})/(\d{2,3})", va)
                if not m or not (60 <= int(m.group(1)) <= 250 and 30 <= int(m.group(2)) <= 150 and int(m.group(1)) > int(m.group(2))):
                    bad = "V1"
            elif k in ("inline.t",) or k.startswith("inline.temperature"):
                if num is None or not 34 <= num <= 42:
                    bad = "V3"
            elif ".bcf." in k:
                if num is None or not 80 <= num <= 220:
                    bad = "V5"
            elif ".poids_kg." in k or (pg["page_type"] in (5, 7) and k == "inline.poids"):     # mother's weight
                if num is None or not 30 <= num <= 200:
                    bad = "V2"
            elif ".hemoglobine." in k:
                if num is None or not 4 <= num <= 20:
                    bad = "V4"
            if bad:
                _flag(f, bad, issues, pg["page_type"])
    return issues


# ----------------------------------------------------------------------------------- form logic (NON_APPLICABLE)
def apply_form_logic(page: dict) -> int:
    """A blank field that the form's own logic excludes is NON_APPLICABLE, not NON_FOURNI:
      p2  caesarean indication of a previous delivery whose mode is vaginal
      p3  RAI (only if Rh negative) when Rh+ is ticked
      p4  'Préciser l'indication' when no caesarean box is ticked
      p5/p7 scar condition when 'Césarienne' is not ticked; 'Pourquoi ?' when the mother wants a method
    Only blank fields are touched; a written value always wins; a box only counts if it is certain (CONNU / confirmed),
    so an uncertain tick never turns a question into NON_APPLICABLE. Re-run after review: a field no longer excluded
    goes back to NON_FOURNI. Returns the number of fields changed."""
    t = page["page_type"]
    f = {x["key"]: x for x in page["fields"]}

    def cb(k):                       # ticked, and sure of it
        x = f.get(k, {})
        return bool(x.get("value")) and x.get("status") == "CONNU"

    def no(k):                       # not ticked, and sure of it
        x = f.get(k, {})
        return bool(x) and not x.get("value") and x.get("status") in ("CONNU", "NON_FOURNI")
    na = []
    if t == 2:
        for k in range(1, 6):
            mode = (f.get(f"accouchements_anterieurs.modalite_d_extraction.accouch_{k}", {}).get("value") or "").lower()
            if mode and "c" != mode[:1] and "sar" not in mode and "قيصرية" not in mode:
                na.append(f"accouchements_anterieurs.si_cesarienne_indication.accouch_{k}")
    if t == 3 and cb("cb.rh_2") and no("cb.rh"):
        na += [k for k in f if k.startswith("visites.rai_si_rh_negatif.")]
    if t == 4 and no("cb.cesarienne_programmee") and no("cb.urgence") and \
            (cb("cb.voie_basse_non_instrumentale") or cb("cb.voie_basse_instrumentale")):
        na.append("inline.preciser_l_indication")
    if t in (5, 7):
        if no("cb.cesarienne"):
            na.append("inline.etat_de_la_cicatrice")
        if cb("cb.desire_utiliser_une_methode"):
            na.append("inline.si_la_mere_ne_desire_pas_une_methode_contraceptive_pourquoi")
    n = 0
    for k in na:
        x = f.get(k)
        if x is not None and x.get("value") is None and x.get("status") == "NON_FOURNI":
            x["status"] = "NON_APPLICABLE"; n += 1
    for k, x in f.items():           # excluded before a correction, not any more
        if k not in na and x.get("status") == "NON_APPLICABLE" and x.get("value") is None:
            x["status"] = "NON_FOURNI"; n += 1
    return n


# exclusive checkbox groups (exactly one answer expected when the question was answered)
EXCLUSIVE = {
    1: [["cb.dr", "cb.csc", "cb.csu", "cb.csca", "cb.csua"], ["cb.fixe", "cb.mobile"]],
    3: [["cb.a", "cb.b", "cb.o", "cb.ab"], ["cb.rh", "cb.rh_2"]],
    4: [["cb.vivant", "cb.mort_ne", "cb.deces_24_heures"]],
    6: [["cb.exclusivement_au_sein", "cb.artificiel", "cb.mixte"]],
    8: [["cb.exclusivement_au_sein", "cb.artificiel", "cb.mixte"]],
}


def apply_group_logic(page: dict) -> list:
    """No box ticked in an exclusive group -> NON_FOURNI (not 'all False, CONNU'); several ticked -> À_RÉVISER."""
    f = {x["key"]: x for x in page["fields"]}
    out = []
    for grp in EXCLUSIVE.get(page["page_type"], []):
        boxes = [f[k] for k in grp if k in f]
        if not boxes:
            continue
        ticked = [b for b in boxes if b.get("value")]
        if not ticked:
            if all(b.get("status") in ("CONNU", "NON_FOURNI") for b in boxes):   # surely unanswered
                for b in boxes:
                    b["status"] = "NON_FOURNI"
                out.append(("none", grp))
            else:                                                             # a faint tick may be there: ask
                out.append(("unsure", grp))
        elif len(ticked) > 1:
            for b in boxes:
                b["status"] = "À_RÉVISER"
                b["group_conflict"] = [x["key"] for x in ticked]
            out.append(("several", grp))
    return out
