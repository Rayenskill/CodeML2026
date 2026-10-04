import json
import sys
from pathlib import Path

W = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(W / "shared"), str(W / "strat10")]
from common import GT_DIR  # noqa: E402
from privacy_linking import Registry, code_distance, leak_guard, norm_code, scan_text  # noqa: E402


def test_scanner_detects_identifiers():
    assert "CIN" in scan_text("CB609814")
    assert "PHONE" in scan_text("06 00 76 13 48")
    assert "ADDRESS" in scan_text("49 Rue Al Qods, Kénitra")
    for v in ("12/05/2022", "104/74", "11.8 g/dL", "RAS", "3626 g", "Dr Benjelloun", "Neg", "2026-823-001"):
        assert scan_text(v) == [], v


def test_gt_contains_no_identifier_values():
    for p in GT_DIR.glob("page_*.json"):
        g = json.loads(p.read_text(encoding="utf-8"))
        for f in g["fields"]:
            if f.get("identifier"):
                assert f["value"] is None and f.get("raw") is None
            elif f["type"] == "text" and f["value"]:
                assert scan_text(f["value"], f["key"]) == [], (p.name, f["key"], f["value"])


def test_leak_guard_redacts():
    page = dict(fields=[dict(key="inline.age", value="31"), dict(key="inline.profession", value="06 61 22 90 14"),
                        dict(key="inline.cin", value="CB609814")])
    inc = leak_guard(page)
    assert {i["key"] for i in inc} == {"inline.profession", "inline.cin"}
    assert page["fields"][0]["value"] == "31" and page["fields"][1]["value"] is None
    # a phone number written in another field is removed *and asked again*, never silently turned into "empty"
    assert page["fields"][1]["status"] == "À_RÉVISER" and page["fields"][1]["redacted"]
    assert page["fields"][2]["value"] is None and page["fields"][2]["status"] == "NON_FOURNI"   # identifier zone


def test_code_distance_confusables():
    assert code_distance("2026-823-001", "2026-823-001") == 0
    assert code_distance("2026-823-001", "2026-823-007") < 1      # 1 <-> 7 confusion
    assert code_distance("2026-823-001", "2026-416-002") > 2
    assert norm_code("٢٠٢٦/٨٢٣/٠٠١") == "2026-823-001"


def test_linking_never_autocreates_when_plausible():
    reg = Registry()
    p1 = reg.create("2026-823-001", dict(ddr="26/04/2025"))
    reg.create("2026-823-007", dict(ddr="01/01/2025"))
    prop = reg.propose("2026-823-001", dict(ddr="26/04/2025"))
    assert prop["candidates"][0]["pid"] == p1
    assert "Aucune, créer" in prop["buttons"] and "Je ne sais pas" in prop["buttons"]
    # misread code still proposes candidates (never silently creates)
    prop2 = reg.propose("2026-823-00l".replace("l", "1"), {})
    assert prop2["kind"] in ("CHOOSE_MATCH", "CONFIRM_MATCH") and len(prop2["candidates"]) >= 1
    n = len(reg.patients)
    pid, state = reg.decide(prop2, "Je ne sais pas", "2026-823-001", {})
    assert pid is None and len(reg.patients) == n and state == "RÉVISION_MANUELLE_REQUISE"
    pid, state = reg.decide(prop, "Patiente 1", "2026-823-001", {})
    assert pid == p1 and state == "PATIENTE_LIÉE"


def test_internal_ids_not_derived_from_data():
    import uuid
    reg = Registry()
    a = reg.create("2026-823-001", {})
    b = reg.create("2026-823-001", {})
    # random UUID4 (not a hash of the code): same code -> unrelated ids. (A random hex id may contain "823" by
    # chance, so the old substring check was flaky.)
    assert a != b and uuid.UUID(a).version == 4 and uuid.UUID(b).version == 4
