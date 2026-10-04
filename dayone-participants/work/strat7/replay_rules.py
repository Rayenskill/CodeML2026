"""Replay the page logic (form / exclusive groups) and the booklet consistency pass on saved raw predictions
(run_eval.py --save_raw 1): rule ablation on the same readings, and (--save_tag) the final predictions/metrics of
the current logic without re-extracting. Strict uncertainty metrics are logged in work/results_strict.md.

python replay_rules.py --raw ~/dayone_local/preds/release3_sev2_raw.pt --model ../../models/crnn_final.pt
python replay_rules.py --raw ~/dayone_local/preds/release3_sev2_raw.pt --model ... --save_tag release3_final --sev 2
"""
import argparse
import copy
import json
import sys
from pathlib import Path

import torch

W = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(W / s) for s in ("shared", "strat2", "strat7", "strat11")]
from common import GT_DIR, LOCAL  # noqa: E402
from eval import load_gt, print_table, score  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--save_tag", default=None, help="save the current rule set's predictions under this tag")
    ap.add_argument("--sev", type=int, default=None)
    a = ap.parse_args()
    import validator
    from extract_zonal import apply_consistency
    from recognizer import Recognizer
    rec = Recognizer(a.model)
    raw = torch.load(Path(a.raw).expanduser(), weights_only=False)
    for page in raw.values():                       # page logic as it runs today (statuses only, values unchanged)
        validator.apply_form_logic(page)
        validator.apply_group_logic(page)
    gts = {k: v for k, v in load_gt(GT_DIR).items() if k in raw}
    res = {}
    for name, visit in (("none", None), ("without R1/H1", False), ("current rules", True)):
        preds = copy.deepcopy(raw)
        issues = []
        if visit is not None:
            validator.VISIT_RULES = visit
            issues = apply_consistency(preds, rec)
        for p in preds.values():
            p.pop("_lp", None)
        r, errors = score(preds, gts)
        res[name] = r["all"]
        print(f"{name:14s} field {r['all']['field_acc']:.4f} filled {r['all']['filled_acc']:.4f} "
              f"status {r['all']['status_acc']:.4f} | repaired {sum(1 for i in issues if i.get('repaired'))} "
              f"flagged {sum(1 for i in issues if not i.get('repaired'))}")
    if a.save_tag:                                   # last pass = current rules
        type_acc = sum(preds[p]["page_type"] == gts[p]["page_type"] for p in gts) / len(gts)
        print_table(r, f"{a.save_tag} sev={a.sev} (replayed rules) type_acc={type_acc:.3f}")
        out = LOCAL / "preds" / f"{a.save_tag}_sev{a.sev}.json"
        out.write_text(json.dumps(dict(preds={str(k): v for k, v in preds.items()}, errors=errors),
                                  ensure_ascii=False), encoding="utf-8")
        x = r["all"]
        with open(W / "results.md", "a", encoding="utf-8") as f:
            f.write(f"| {a.save_tag} | {a.sev} | 1-80 | {type_acc:.3f} | {x['field_acc']:.4f} | {x['filled_acc']:.4f} | "
                    f"{x['blank_acc']:.4f} | {x['checkbox_acc']:.4f} | {x['status_acc']:.4f} | {x['ece']:.4f} |\n")
        strict = W / "results_strict.md"
        if not strict.exists():
            strict.write_text("# Strict uncertainty metrics (work/shared/eval.py, written by strat7/replay_rules.py)\n\n"
                              "| tag | severity | auto-accepted (CONNU) | wrong among CONNU | wrong among CONNU (text) | "
                              "written values declared blank | questions / page | ECE text | ECE checkboxes |\n"
                              "|---|---|---|---|---|---|---|---|---|\n", encoding="utf-8")
        with open(strict, "a", encoding="utf-8") as f:
            f.write(f"| {a.save_tag} | {a.sev} | {x['auto_rate']:.3f} | {x['connu_err']:.4f} | {x['connu_err_text']:.4f} | "
                    f"{x['missed_filled']:.4f} | {x['questions_per_page']:.1f} | {x['ece_text']:.4f} | "
                    f"{x['ece_checkbox']:.4f} |\n")
    return res


if __name__ == "__main__":
    main()
