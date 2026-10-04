"""Strategy 7 tests: T1 no false alarms on the 10 GT booklets; T2/T3 injected OCR-like date errors are caught and
repaired when the recogniser finds the implied value plausible (stub scorer: edit-distance based)."""
import copy
import json
import random
import sys
from pathlib import Path

W = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(W / "shared"), str(W / "strat7")]
from common import GT_DIR  # noqa: E402
from validator import validate_booklet  # noqa: E402


def booklets():
    pages = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(GT_DIR.glob("page_*.json"))]
    out = {}
    for g in pages:
        g = copy.deepcopy(g)
        for f in g["fields"]:
            f["confidence"] = 0.95
        out.setdefault(g["patient"], []).append(g)
    return out


class EditScorer:
    """Stands in for the CRNN: log-likelihood falls with the edit distance to what is 'written'."""
    def __init__(self, truth):
        self.truth = truth

    def score(self, lp, text):
        from rapidfuzz.distance import Levenshtein
        return -3.0 * Levenshtein.distance(self.truth[lp], text or "")


def test_no_false_alarms_on_gt():
    n_issues = 0
    for pid, pages in booklets().items():
        iss = validate_booklet(pages)
        n_issues += len(iss)
        assert not iss, (pid, iss)
    assert n_issues == 0


def test_injected_date_errors_repaired():
    random.seed(0)
    caught = repaired = total = 0
    targets = [(3, "inline.date_prevue_d_accouchement"), (3, "inline.date_de_depassement_de_terme"),
               (4, "inline.age_gestationnel"), (6, "inline.age"), (6, "inline.date_de_la_consultation")]
    for pid, pages in booklets().items():
        for t, key in targets:
            pg = copy.deepcopy(pages)
            truth = {}
            for p in pg:
                p["_lp"] = {}
                for f in p["fields"]:
                    if f["type"] == "text" and f.get("value"):
                        k = (p["page_type"], f["key"]); truth[k] = f["value"]; p["_lp"][f["key"]] = k
            p = next(p for p in pg if p["page_type"] == t)
            f = next(f for f in p["fields"] if f["key"] == key)
            orig = f["value"]
            v = list(orig)
            i = random.choice([j for j, c in enumerate(v) if c.isdigit()])
            v[i] = {"1": "7", "7": "1", "3": "8", "8": "3", "0": "6", "6": "0", "5": "6", "4": "9", "9": "4", "2": "7"}[v[i]]
            f["value"] = "".join(v)
            f["confidence"] = 0.6
            iss = validate_booklet(pg, EditScorer(truth))
            total += 1
            caught += any(x["key"] == key for x in iss)
            repaired += f["value"] == orig
    print(f"caught {caught}/{total}, repaired {repaired}/{total}")
    assert caught / total >= 0.8 and repaired / total >= 0.7


def _with_truth(pages):
    truth = {}
    for p in pages:
        p["_lp"] = {}
        for f in p["fields"]:
            if f["type"] == "text" and f.get("value"):
                k = (p["page_type"], f["key"]); truth[k] = f["value"]; p["_lp"][f["key"]] = k
    return truth


def test_visit_table_soft_rules():
    """R1 (appointment = visit + 28 d) and H1 (fundal height = SA - 4): an OCR digit error in any of the three
    columns is repaired from the other one; a midwife-confirmed value is never touched; soft rules never flag."""
    random.seed(1)
    swap = {"1": "7", "7": "1", "3": "8", "8": "3", "0": "6", "6": "0", "5": "6", "4": "9", "9": "4", "2": "7"}
    total = repaired = 0
    for pid, pages in booklets().items():
        p3 = next(p for p in pages if p["page_type"] == 3)
        cols = [f["key"].split(".")[-1] for f in p3["fields"]
                if f["key"].startswith("visites.venue_le.") and f.get("value")]
        for col in cols[:3]:
            for field in ("rendez_vous", "venue_le", "hu_cm"):
                pg = copy.deepcopy(pages)
                truth = _with_truth(pg)
                f = next(f for f in next(p for p in pg if p["page_type"] == 3)["fields"]
                         if f["key"] == f"visites.{field}.{col}")
                if not f.get("value"):
                    continue
                orig = f["value"]; v = list(orig)
                i = random.choice([j for j, c in enumerate(v) if c.isdigit()])
                v[i] = swap[v[i]]
                f["value"] = "".join(v)
                validate_booklet(pg, EditScorer(truth))
                total += 1; repaired += f["value"] == orig
                f["value"], f["reviewed"], f["status"] = "".join(v), True, "CONNU"     # confirmed by the midwife
                validate_booklet(pg, EditScorer(truth))
                assert f["value"] == "".join(v) and f["status"] == "CONNU"
    print(f"visit rules repaired {repaired}/{total}")
    assert total >= 60 and repaired / total >= 0.85


def test_uncertain_boxes_keep_their_doubt():
    """No tick read in an exclusive group is NON_FOURNI only if every box is certain; an uncertain box never makes
    a field NON_APPLICABLE, and the form logic follows the midwife's corrections."""
    from validator import apply_form_logic, apply_group_logic
    page = {"page_type": 5, "fields": [
        {"key": "cb.cesarienne", "type": "checkbox", "value": False, "status": "À_RÉVISER"},
        {"key": "inline.etat_de_la_cicatrice", "type": "text", "value": None, "status": "NON_FOURNI"}]}
    apply_form_logic(page)
    assert page["fields"][1]["status"] == "NON_FOURNI"            # uncertain box: the scar question stays open
    page["fields"][0]["status"] = "CONNU"                         # midwife: no caesarean
    apply_form_logic(page)
    assert page["fields"][1]["status"] == "NON_APPLICABLE"
    page["fields"][0].update(value=True)                          # she corrects: there was a caesarean
    apply_form_logic(page)
    assert page["fields"][1]["status"] == "NON_FOURNI"
    grp = {"page_type": 3, "fields": [{"key": k, "type": "checkbox", "value": False, "status": s}
                                      for k, s in (("cb.rh", "CONNU"), ("cb.rh_2", "À_RÉVISER"))]}
    apply_group_logic(grp)
    assert [f["status"] for f in grp["fields"]] == ["CONNU", "À_RÉVISER"]
    grp["fields"][1]["status"] = "CONNU"
    apply_group_logic(grp)
    assert [f["status"] for f in grp["fields"]] == ["NON_FOURNI", "NON_FOURNI"]
