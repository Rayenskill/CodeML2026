"""Run the zonal extractor on specimen pages (clean or degraded) and score against the GT.

python run_eval.py --model crnn_v1.pt [--omr omr_v1.pt] --sev 0 2 4 --pages 1-80 [--true_type] [--tag v1]
Writes ~/dayone_local/preds/<tag>_sev<k>.json and appends a markdown summary to work/results.md.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

W = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(W / "shared"), str(W / "strat2"), str(W / "strat11"), str(W / "strat18"), str(W / "strat6")]
from common import GT_DIR, LOCAL  # noqa: E402
from eval import load_gt, print_table, score  # noqa: E402


def parse_pages(s):
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")     # Arabic values in logs on a cp1252 console
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--omr", default=None)
    ap.add_argument("--cal", default=None)
    ap.add_argument("--sev", nargs="+", type=int, default=[0])
    ap.add_argument("--pages", default="1-80")
    ap.add_argument("--true_type", action="store_true")
    ap.add_argument("--delta", type=float, default=6.0)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--consistency", type=int, default=1)
    ap.add_argument("--tta", type=int, default=0)
    ap.add_argument("--views", default="1.0,1.2,1.4")
    ap.add_argument("--grammar", type=int, default=1, help="grammar-constrained beam search (strategy 7)")
    ap.add_argument("--save_raw", type=int, default=0, help="also save predictions + rule log-probs before the "
                                                             "consistency pass (replay rules without re-extracting)")
    a = ap.parse_args()
    from extract_zonal import Extractor, run_specimen
    import fieldlogic
    fieldlogic.GRAMMAR_BEAM = bool(a.grammar)
    from recognizer import Recognizer
    if "," in a.model:
        from recognizer import EnsembleRecognizer
        rec = EnsembleRecognizer(a.model.split(","))
    else:
        rec = Recognizer(a.model)
    omr = None
    if a.omr:
        from omr import OMR
        omr = OMR(a.omr)
    cal = None
    if a.cal:
        from calibrate import Calibrator
        cal = Calibrator.load(a.cal)
    ext = Extractor(rec, omr, cal, delta=a.delta, tta=bool(a.tta), views=[float(x) for x in a.views.split(",")])
    gts = load_gt(GT_DIR)
    pages = parse_pages(a.pages)
    gts = {k: v for k, v in gts.items() if k in pages}
    (LOCAL / "preds").mkdir(exist_ok=True)
    for sev in a.sev:
        t0 = time.time()
        preds = run_specimen(ext, pages, sev, seed=a.seed + sev, use_true_type=a.true_type)
        if a.save_raw:
            import copy
            import torch
            torch.save(copy.deepcopy(preds), LOCAL / "preds" / f"{a.tag}_sev{sev}_raw.pt")
        if a.consistency:
            from extract_zonal import apply_consistency
            res0, _ = score(preds, gts)
            print(f"before consistency: field_acc {res0['all']['field_acc']:.4f} filled_acc {res0['all']['filled_acc']:.4f}")
            issues = apply_consistency(preds, rec)
            print(f"consistency issues {len(issues)}, repaired {sum(1 for i in issues if i.get('repaired'))}")
        for p in preds.values():
            p.pop("_lp", None)
        res, errors = score(preds, gts)
        type_acc = sum(preds[p]["page_type"] == gts[p]["page_type"] for p in pages) / len(pages)
        title = f"{a.tag} sev={sev} pages={a.pages} type_acc={type_acc:.3f} ({time.time() - t0:.0f}s)"
        print_table(res, title)
        (LOCAL / "preds" / f"{a.tag}_sev{sev}.json").write_text(json.dumps(dict(preds={str(k): v for k, v in preds.items()},
                                                                                 errors=errors), ensure_ascii=False), encoding="utf-8")
        with open(W / "results.md", "a", encoding="utf-8") as f:
            r = res["all"]
            f.write(f"| {a.tag} | {sev} | {a.pages} | {type_acc:.3f} | {r['field_acc']:.4f} | {r['filled_acc']:.4f} | "
                    f"{r['blank_acc']:.4f} | {r['checkbox_acc']:.4f} | {r['status_acc']:.4f} | {r['ece']:.4f} |\n")
        print("sample errors:", errors[:25])


if __name__ == "__main__":
    main()
