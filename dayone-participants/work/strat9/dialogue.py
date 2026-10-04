"""Strategy 9: deterministic conversational review (WhatsApp-style), FR / EN.

The agent summarises what it read, then asks ONLY about uncertain fields (lowest confidence first,
clinically important fields before others), always showing the evidence crop and quick replies:
  [Confirmer] [Corriger] [Reprendre la photo] (+ candidate values as buttons when the validator proposed one)
Manual entry is available for every field when AI is unavailable. Multi-page sessions: one document per
booklet; a re-photographed page shows the stored record and lets the midwife choose what to update.
"""
from __future__ import annotations

MSG = {
    "fr": dict(summary="J'ai lu la page « {page} » : {n_ok} champs sûrs, {n_q} à vérifier, {n_blank} vides.",
               ask="Je ne suis pas sûre de « {label} » : j'ai lu « {value} » ({conf:.0%}). C'est correct ?",
               ask_blank="La case « {label} » semble illisible. Que faut-il saisir ?",
               confirm="Confirmer", correct="Corriger", retake="Reprendre la photo", manual="Saisie manuelle",
               type_value="Écrivez la valeur pour « {label} » (ou « vide », « inconnu »).",
               done="Merci, la page est validée ({n} champs).", retake_ack="D'accord, reprenez la photo de la page.",
               next_page="Page suivante ? Envoyez la photo ou tapez « fin ».",
               match="Cette visite correspond-elle à une patiente existante ?",
               ai_down="L'IA n'est pas disponible : nous passons en saisie manuelle, champ par champ."),
    "en": dict(summary="I read page “{page}”: {n_ok} sure fields, {n_q} to check, {n_blank} empty.",
               ask="I'm not sure about “{label}”: I read “{value}” ({conf:.0%}). Is that right?",
               ask_blank="The box “{label}” looks illegible. What should I enter?",
               confirm="Confirm", correct="Edit", retake="Retake photo", manual="Manual entry",
               type_value="Type the value for “{label}” (or “empty”, “unknown”).",
               done="Thanks, the page is validated ({n} fields).", retake_ack="OK, please retake the photo of the page.",
               next_page="Next page? Send the photo or type “end”.",
               match="Does this visit belong to an existing patient?",
               ai_down="AI is unavailable: switching to manual entry, field by field."),
}
PRIORITY = ("ta", "hemoglobine", "serologie_vih", "syphilis", "ddr", "date_prevue", "poids", "date_de_l_accouchement",
            "t", "pouls", "bcf")


def label_of(key: str) -> str:
    parts = key.split(".")
    if parts[0] == "inline":
        return parts[1].replace("_", " ")
    if parts[0] == "cb":
        return parts[1].replace("_", " ")
    return " / ".join(p.replace("_", " ") for p in parts[1:])


class Review:
    """State machine for one page review. Feed user replies with .reply(text); read .outbox for messages."""

    def __init__(self, page: dict, page_name: str, lang="fr", tau_ask=0.9, ai_available=True):
        self.page, self.lang, self.m = page, lang, MSG[lang]
        self.outbox: list[dict] = []
        self.done = False
        self.retake = False
        fields = page["fields"]
        if not ai_available:
            self.queue = [f for f in fields if f["type"] == "text"]
            self.mode = "manual"
            self._say(self.m["ai_down"])
        else:
            unsure = [f for f in fields if f.get("status") in ("À_RÉVISER", "ILLISIBLE") or
                      (f.get("value") is not None and f.get("confidence", 1) < tau_ask)]
            unsure.sort(key=lambda f: (not any(p in f["key"] for p in PRIORITY), f.get("confidence", 0)))
            self.queue = unsure
            self.mode = "ai"
            n_blank = sum(1 for f in fields if f["type"] == "text" and f.get("value") is None)
            self._say(self.m["summary"].format(page=page_name, n_ok=len(fields) - len(unsure) - n_blank,
                                               n_q=len(unsure), n_blank=n_blank))
        self.cur = None
        self._next()

    def _say(self, text, buttons=None, evidence=None):
        self.outbox.append(dict(text=text, buttons=buttons or [], evidence=evidence))

    def _next(self):
        if not self.queue:
            self.done = True
            for f in self.page["fields"]:
                if f.get("status") in ("À_RÉVISER", "ILLISIBLE") and not f.get("reviewed"):
                    f["status"] = "CONNU" if f.get("value") else "NON_FOURNI"
            self._say(self.m["done"].format(n=len(self.page["fields"])))
            return
        self.cur = self.queue.pop(0)
        f, lab = self.cur, label_of(self.cur["key"])
        if self.mode == "manual":
            self._say(self.m["type_value"].format(label=lab), evidence=f.get("bbox"))
            self.awaiting = "value"
            return
        extra = [s for s in f.get("suggestions", []) if s != f.get("value")][:2]
        if f.get("value") is None or f.get("status") == "ILLISIBLE":
            self._say(self.m["ask_blank"].format(label=lab), [self.m["manual"], self.m["retake"]] + extra, f.get("bbox"))
        else:
            self._say(self.m["ask"].format(label=lab, value=f["value"], conf=f.get("confidence", 0)),
                      [self.m["confirm"], self.m["correct"], self.m["retake"]] + extra, f.get("bbox"))
        self.awaiting = "choice"

    def reply(self, text: str):
        if self.done:
            return
        f = self.cur
        t = text.strip()
        if self.awaiting == "choice":
            if t == self.m["confirm"]:
                f.update(status="CONNU", reviewed="confirmed", confidence=1.0)
            elif t in (self.m["correct"], self.m["manual"]):
                self._say(self.m["type_value"].format(label=label_of(f["key"])))
                self.awaiting = "value"
                return
            elif t == self.m["retake"]:
                self.retake, self.done = True, True
                self._say(self.m["retake_ack"])
                return
            else:                                   # a suggestion button or a typed value
                self._set(f, t)
        else:
            self._set(f, t)
        self._next()

    def _set(self, f, t):
        low = t.lower()
        if low in ("vide", "empty", "-", "–"):
            f.update(value=None, status="NON_FOURNI")
        elif low in ("inconnu", "unknown", "?"):
            f.update(value=t, status="INCONNU")
        else:
            f.update(value=t, status="CONNU")
        f.update(reviewed="corrected", confidence=1.0)
