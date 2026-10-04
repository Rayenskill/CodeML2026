"""Strategy 13 simulation on real outputs: validated record = GT of the earlier visit; new capture = the
severity-2 extraction of the same page (saved by run_eval --tag final). Also simulates the next visit by
blanking later visit columns in the validated record (new column = normal extraction)."""
import json
import sys
from pathlib import Path

W = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(W / s) for s in ("shared", "strat5", "strat13")]
from common import GT_DIR, LOCAL  # noqa: E402
from reconcile import reconcile  # noqa: E402
from vocab import canonical  # noqa: E402


def run(tag="final_sev2", visit_cut=("t3_m8", "t3_m9")):
    p = LOCAL / "preds" / f"{tag}.json"
    if not p.exists():
        import pytest
        pytest.skip("no saved predictions")
    preds = json.loads(p.read_text(encoding="utf-8"))["preds"]
    tot = dict(n=0, ok_without=0, ok_with=0, q_without=0, q_with=0)
    for pid, pr in preds.items():
        g = json.loads((GT_DIR / f"page_{int(pid):02d}.json").read_text(encoding="utf-8"))
        if pr["page_type"] != g["page_type"]:
            continue
        # validated record of the previous visit: GT with the most recent visit columns still empty
        val = dict(fields=[dict(f, value=(None if any(c in f["key"] for c in visit_cut) else f["value"]))
                           for f in g["fields"] if f["type"] == "text" and not f.get("identifier")])
        merged, st = reconcile(val, dict(pr, fields=[f for f in pr["fields"] if f["type"] == "text"]))
        gt = {f["key"]: f["value"] for f in g["fields"]}
        for a, b in zip([f for f in pr["fields"] if f["type"] == "text"], merged["fields"]):
            if a["key"] not in gt:
                continue
            tot["n"] += 1
            tot["ok_without"] += canonical(a["value"]) == canonical(gt[a["key"]])
            tot["ok_with"] += canonical(b["value"]) == canonical(gt[a["key"]])
            tot["q_without"] += a.get("status") in ("À_RÉVISER", "ILLISIBLE")
            tot["q_with"] += b.get("status") in ("À_RÉVISER", "ILLISIBLE")
    n_pages = len(preds)
    print(f"accuracy without {tot['ok_without'] / tot['n']:.4f} -> with reconciliation {tot['ok_with'] / tot['n']:.4f}; "
          f"questions/page {tot['q_without'] / n_pages:.1f} -> {tot['q_with'] / n_pages:.1f}")
    return tot


def test_reconciliation_improves_and_never_overwrites_silently():
    tot = run()
    assert tot["ok_with"] >= tot["ok_without"]
    # injected paper correction on a stable field is surfaced, not overwritten
    from reconcile import reconcile as rc
    m, st = rc(dict(fields=[dict(key="inline.ddr", value="26/04/2025")]),
               dict(fields=[dict(key="inline.ddr", value="14/03/2025", status="CONNU", confidence=0.95)]))
    assert m["fields"][0]["status"] == "À_RÉVISER" and m["fields"][0]["previous"] == "26/04/2025"


if __name__ == "__main__":
    run()
