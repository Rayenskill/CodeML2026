"""Shared scorer for every extraction strategy.

score(preds, gts) where both are {page_id: page_dict}; page_dict = {"page_type", "fields": [...]}
Each predicted field: {key, value, status, confidence}. Identifier fields are never scored (and any
predicted value for them counts as a leak).

Metrics
  text_acc       : all text fields, canonical(pred) == canonical(gt) (blank == blank counts)
  filled_acc     : text fields whose GT is filled
  blank_acc      : text fields whose GT is blank (NON_FOURNI)  -> 1 - hallucination rate
  checkbox_acc   : checkbox fields
  field_acc      : all fields (text + checkbox)  <- headline number
  status_acc     : predicted status == GT status (text fields)
  ece            : calibration of `confidence` w.r.t. correctness
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "strat5"))
from vocab import canonical  # noqa: E402


def norm(v):
    c = canonical(v)
    return c


def ece(conf, correct, bins=10):
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    if len(conf) == 0:
        return float("nan")
    e = 0.0
    for lo in np.linspace(0, 1, bins, endpoint=False):
        m = (conf >= lo) & (conf < lo + 1 / bins + (1e-9 if lo + 1 / bins >= 1 else 0))
        if m.any():
            e += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(e)


def score(preds: dict, gts: dict, by=("page_type",)):
    agg = collections.defaultdict(lambda: collections.Counter())
    confs, corrs = [], []
    tconfs, tcorrs, cconfs, ccorrs = [], [], [], []      # text / checkbox calibration, reported separately
    questions = 0
    errors = []
    leaks = 0
    for pid, g in gts.items():
        p = preds.get(pid, {"fields": []})
        pf = {f["key"]: f for f in p["fields"]}
        questions += sum(f.get("status") in ("À_RÉVISER", "ILLISIBLE") for f in p["fields"])
        for f in g["fields"]:
            key = f["key"]
            pr = pf.get(key)
            groups = ["all", f"type{g['page_type']}"]
            if f.get("identifier") or f.get("type") == "identifier":
                if pr and pr.get("value"):
                    leaks += 1
                continue
            if f["type"] == "checkbox":
                ok = pr is not None and bool(pr.get("value")) == bool(f["value"])
                for gr in groups:
                    agg[gr]["cb_n"] += 1; agg[gr]["cb_ok"] += ok
                    agg[gr]["n"] += 1; agg[gr]["ok"] += ok
                if pr is not None and pr.get("confidence") is not None:
                    confs.append(pr["confidence"]); corrs.append(ok)
                    cconfs.append(pr["confidence"]); ccorrs.append(ok)
                if pr is not None and pr.get("status") == "CONNU":
                    for gr in groups:
                        agg[gr]["k_n"] += 1; agg[gr]["k_bad"] += not ok
                if not ok:
                    errors.append((pid, key, f["value"], pr and pr.get("value")))
                continue
            gv = norm(f["value"])
            pv = norm(pr.get("value")) if pr else None
            ok = gv == pv
            filled = gv is not None
            for gr in groups:
                agg[gr]["n"] += 1; agg[gr]["ok"] += ok
                agg[gr]["t_n"] += 1; agg[gr]["t_ok"] += ok
                if filled:
                    agg[gr]["f_n"] += 1; agg[gr]["f_ok"] += ok
                    # a written value read as blank and declared NON_FOURNI: never shown to the midwife
                    agg[gr]["miss"] += pv is None and bool(pr) and pr.get("status") == "NON_FOURNI"
                else:
                    agg[gr]["b_n"] += 1; agg[gr]["b_ok"] += ok
                if pr and pr.get("status"):
                    gs = f["status"]
                    ps = pr["status"]
                    # a correct, confirmed-later value is CONNU; À_RÉVISER on a correct value is still acceptable
                    st_ok = ps == gs or (gs == "CONNU" and ps == "À_RÉVISER")
                    agg[gr]["s_n"] += 1; agg[gr]["s_ok"] += st_ok
            if pr is not None and pr.get("confidence") is not None:
                confs.append(pr["confidence"]); corrs.append(ok)
                tconfs.append(pr["confidence"]); tcorrs.append(ok)
            if pr is not None and pr.get("status") == "CONNU":
                for gr in groups:
                    agg[gr]["k_n"] += 1; agg[gr]["k_bad"] += not ok
                    agg[gr]["kt_n"] += 1; agg[gr]["kt_bad"] += not ok
            if not ok:
                errors.append((pid, key, f["value"], pr and pr.get("value")))
    out = {}
    for gr, c in agg.items():
        out[gr] = dict(field_acc=c["ok"] / max(c["n"], 1), text_acc=c["t_ok"] / max(c["t_n"], 1),
                       filled_acc=c["f_ok"] / max(c["f_n"], 1), blank_acc=c["b_ok"] / max(c["b_n"], 1),
                       checkbox_acc=c["cb_ok"] / max(c["cb_n"], 1), status_acc=c["s_ok"] / max(c["s_n"], 1),
                       # strict uncertainty metrics: share of auto-accepted (CONNU) fields, how many of them are
                       # wrong, and written values silently declared blank
                       auto_rate=c["k_n"] / max(c["n"], 1), connu_err=c["k_bad"] / max(c["k_n"], 1),
                       connu_err_text=c["kt_bad"] / max(c["kt_n"], 1),
                       missed_filled=c["miss"] / max(c["f_n"], 1), n=c["n"], n_filled=c["f_n"])
    out["all"]["ece"] = ece(confs, corrs)
    out["all"]["ece_text"] = ece(tconfs, tcorrs)
    out["all"]["ece_checkbox"] = ece(cconfs, ccorrs)
    out["all"]["questions_per_page"] = questions / max(len(gts), 1)
    out["all"]["leaks"] = leaks
    return out, errors


def load_gt(gt_dir):
    """GT pages; blank fields excluded by the form's logic get status NON_APPLICABLE (strategy 7 rules)."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "strat7"))
    from validator import apply_form_logic
    out = {}
    for p in sorted(Path(gt_dir).glob("page_*.json")):
        g = json.loads(p.read_text(encoding="utf-8"))
        apply_form_logic(g)
        out[int(p.stem.split("_")[1])] = g
    return out


def print_table(res, title=""):
    print(f"### {title}")
    print("| group | field_acc | text_acc | filled_acc | blank_acc | checkbox_acc | status_acc | n |")
    print("|---|---|---|---|---|---|---|---|")
    for gr in sorted(res, key=lambda g: (g != "all", g)):
        r = res[gr]
        print(f"| {gr} | {r['field_acc']:.4f} | {r['text_acc']:.4f} | {r['filled_acc']:.4f} | {r['blank_acc']:.4f} | "
              f"{r['checkbox_acc']:.4f} | {r['status_acc']:.4f} | {r['n']} |")
    if "ece" in res.get("all", {}):
        a = res["all"]
        print(f"ECE={a['ece']:.4f} (text {a['ece_text']:.4f}, checkbox {a['ece_checkbox']:.4f}) leaks={a['leaks']} | "
              f"auto-accepted {a['auto_rate']:.3f}, wrong among CONNU {a['connu_err']:.4f} (text {a['connu_err_text']:.4f}), "
              f"written values declared blank {a['missed_filled']:.4f}, questions/page {a['questions_per_page']:.1f}")
