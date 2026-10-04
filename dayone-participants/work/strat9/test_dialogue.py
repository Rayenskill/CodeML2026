"""Golden-transcript tests for the review dialogue (strategy 9)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dialogue import Review  # noqa: E402


def page():
    return dict(fields=[
        dict(key="inline.ddr", type="text", value="26/04/2025", status="CONNU", confidence=0.99),
        dict(key="visites.ta.t1_v2", type="text", value="104/74", status="À_RÉVISER", confidence=0.6,
             suggestions=["104/76"]),
        dict(key="visites.poids_kg.t1_v2", type="text", value=None, status="ILLISIBLE", confidence=0.2),
        dict(key="visites.hu_cm.t1_v2", type="text", value=None, status="NON_FOURNI", confidence=0.97),
    ])


def test_asks_only_uncertain_fields_priority_first():
    r = Review(page(), "Grossesse actuelle")
    assert "2 à vérifier" in r.outbox[0]["text"]
    # lowest confidence first among priority fields: the illegible weight, then the blood pressure
    assert "poids" in r.outbox[1]["text"] and r.outbox[1]["buttons"][0] == "Saisie manuelle"
    r.reply("Saisie manuelle"); r.reply("58.8")
    q = r.outbox[-1]
    assert "ta" in q["text"] and "104/74" in q["text"]
    assert q["buttons"][:3] == ["Confirmer", "Corriger", "Reprendre la photo"]
    assert "104/76" in q["buttons"]                     # validator suggestion as a quick reply


def test_confirm_correct_manual_flow():
    p = page()
    r = Review(p, "Grossesse actuelle")
    r.reply("Saisie manuelle"); r.reply("58.8")
    assert p["fields"][2]["value"] == "58.8"
    r.reply("Corriger"); r.reply("104/76")
    assert p["fields"][1]["value"] == "104/76" and p["fields"][1]["status"] == "CONNU"
    assert r.done and "validée" in r.outbox[-1]["text"]
    assert all(f["status"] in ("CONNU", "NON_FOURNI") for f in p["fields"])


def test_retake_and_unknown():
    p = page()
    r = Review(p, "x")
    r.reply("Reprendre la photo")
    assert r.retake and r.done
    p = page()
    r = Review(p, "x", lang="en")
    r.reply("Manual entry"); r.reply("unknown")
    assert p["fields"][2]["status"] == "INCONNU"


def test_manual_mode_when_ai_down():
    p = page()
    r = Review(p, "x", ai_available=False)
    assert "manuelle" in r.outbox[0]["text"]
    for v in ("26/04/2025", "104/74", "vide", "12"):
        r.reply(v)
    assert r.done and p["fields"][2]["status"] == "NON_FOURNI"
