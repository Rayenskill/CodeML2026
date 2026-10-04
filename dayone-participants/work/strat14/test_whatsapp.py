import sys
from pathlib import Path

W = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(W / "strat14"), str(W / "strat9")]
from dialogue import Review  # noqa: E402
from whatsapp_adapter import Webhook, to_payloads  # noqa: E402


def test_payload_shapes():
    p = to_payloads("2126000", dict(text="Je ne suis pas sûre…", buttons=["Confirmer", "Corriger", "Reprendre la photo"]))
    assert p[0]["interactive"]["type"] == "button" and len(p[0]["interactive"]["action"]["buttons"]) == 3
    assert all(len(b["reply"]["title"]) <= 20 for b in p[0]["interactive"]["action"]["buttons"])
    p = to_payloads("2126000", dict(text="Correspondance ?", buttons=["Patiente 1", "Patiente 2", "Aucune, créer", "Je ne sais pas"]))
    assert p[0]["interactive"]["type"] == "list"
    p = to_payloads("2126000", dict(text="ok"), media_id="m1")
    assert p[0]["type"] == "image" and p[1]["type"] == "text"


def test_webhook_routes_buttons_and_dedupes():
    sent = []
    page = dict(fields=[dict(key="visites.ta.t1_v2", type="text", value="104/74", status="À_RÉVISER", confidence=0.6)])
    dlg = Review(page, "Grossesse actuelle")
    wh = Webhook(sent.append, lambda u: dlg)
    wh.last_buttons["212600"] = dlg.outbox[-1]["buttons"]
    ev = {"entry": [{"changes": [{"value": {"messages": [{"id": "wamid.1", "from": "212600", "type": "interactive",
           "interactive": {"type": "button_reply", "button_reply": {"id": "b0", "title": "Confirmer"}}}]}}]}]}
    wh.handle(ev); wh.handle(ev)                       # Meta retry: processed once
    assert page["fields"][0]["status"] == "CONNU" and dlg.done
    assert len(sent) == 1 and "validée" in sent[0]["text"]["body"]
