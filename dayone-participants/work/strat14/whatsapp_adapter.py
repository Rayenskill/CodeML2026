"""Strategy 14: WhatsApp Business Platform (Cloud API) adapter for the strategy 9 dialogue manager.

Only the adapter is implemented and tested offline (mocked Graph API): a real sandbox needs a Meta test
number and access token (not available here) and, for real patients, an approved data-processing setup —
the brief forbids sending real patient data to a third party. Synthetic data only.

* dialogue message -> Cloud API payload: text, image (evidence crop), interactive reply buttons
  (max 3, titles <= 20 chars) or an interactive list when there are more options (patient matching: 4 options)
* webhook: dedupe by message id (Meta retries deliveries), route button replies back to the dialogue
"""
from __future__ import annotations


def to_payloads(to: str, msg: dict, media_id: str | None = None) -> list[dict]:
    out = []
    if media_id:
        out.append({"messaging_product": "whatsapp", "to": to, "type": "image", "image": {"id": media_id}})
    buttons = msg.get("buttons") or []
    if not buttons:
        out.append({"messaging_product": "whatsapp", "to": to, "type": "text", "text": {"body": msg["text"][:4096]}})
    elif len(buttons) <= 3:
        out.append({"messaging_product": "whatsapp", "to": to, "type": "interactive", "interactive": {
            "type": "button", "body": {"text": msg["text"][:1024]},
            "action": {"buttons": [{"type": "reply", "reply": {"id": f"b{i}", "title": b[:20]}}
                                   for i, b in enumerate(buttons)]}}})
    else:
        out.append({"messaging_product": "whatsapp", "to": to, "type": "interactive", "interactive": {
            "type": "list", "body": {"text": msg["text"][:1024]},
            "action": {"button": "Choisir", "sections": [{"title": "Options", "rows": [
                {"id": f"b{i}", "title": b[:24]} for i, b in enumerate(buttons[:10])]}]}}})
    return out


class Webhook:
    """Routes incoming WhatsApp events to a per-user dialogue; idempotent on message ids."""

    def __init__(self, send, dialogue_for):
        self.send, self.dialogue_for = send, dialogue_for
        self.seen: set[str] = set()
        self.last_buttons: dict[str, list[str]] = {}

    def handle(self, event: dict):
        for entry in event.get("entry", []):
            for ch in entry.get("changes", []):
                for m in ch.get("value", {}).get("messages", []):
                    if m["id"] in self.seen:
                        continue
                    self.seen.add(m["id"])
                    user = m["from"]
                    if m["type"] == "interactive":
                        it = m["interactive"]
                        rid = (it.get("button_reply") or it.get("list_reply"))["id"]
                        text = self.last_buttons.get(user, [])[int(rid[1:])]
                    elif m["type"] == "text":
                        text = m["text"]["body"]
                    elif m["type"] == "image":
                        text = "__IMAGE__:" + m["image"]["id"]
                    else:
                        continue
                    dlg = self.dialogue_for(user)
                    n0 = len(dlg.outbox)
                    dlg.reply(text)
                    for msg in dlg.outbox[n0:]:
                        self.last_buttons[user] = msg.get("buttons") or []
                        for p in to_payloads(user, msg):
                            self.send(p)
