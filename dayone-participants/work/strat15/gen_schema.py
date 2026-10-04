"""Build app/schema.json (field list + readable FR/EN labels per page type) for the phone app, so review
labels and full manual entry work offline."""
import json
import re
import sys
from pathlib import Path

W = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(W / "shared"))
from common import IDENTIFIER_LABELS, SHARED  # noqa: E402

T = json.loads((SHARED / "templates.json").read_text(encoding="utf-8"))
NAMES = {1: ("Fiche de surveillance", "Cover sheet"), 2: ("Identification et antécédents", "Identification & history"),
         3: ("Grossesse actuelle", "Current pregnancy"), 4: ("Accouchement", "Delivery"),
         5: ("Post-partum précoce · mère", "Early postpartum · mother"),
         6: ("Post-partum précoce · nouveau-né", "Early postpartum · newborn"),
         7: ("Post-partum tardif · mère", "Late postpartum · mother"),
         8: ("Post-partum tardif · nouveau-né", "Late postpartum · newborn")}
COLS = {"t1_v1": "1er trim. V1", "t1_v2": "1er trim. V2", "t1_v3": "1er trim. V3", "t2_v1": "2e trim. V1",
        "t2_v2": "2e trim. V2", "t2_v3": "2e trim. V3", "t3_m7": "7e mois", "t3_m8": "8e mois", "t3_m9": "9e mois"}
NICE = {"ta": "TA", "hu_cm": "HU (cm)", "bcf": "BCF", "poids_kg": "Poids (kg)", "ddr": "DDR", "t": "T°",
        "rai_si_rh_negatif": "RAI", "n_de_la_fiche": "N° de la fiche (code)", "le": "Rubéole · le", "le_2": "Hépatite B · le"}


def human(s):
    s = re.sub(r"_\d+$", lambda m: f" ({m.group(0)[1:]})", s)
    return NICE.get(s, s.replace("_", " ").strip().capitalize())


def label(key, z):
    p = key.split(".")
    if p[0] == "cb":
        g = (z.get("group") or "").rstrip(" :")
        lab = z.get("label") or human(p[1])
        return f"{g} · {lab}" if g and g.lower() != lab.lower() else lab
    if p[0] == "inline":
        return human(p[1])
    row, col = p[1], p[2]
    col = COLS.get(col, col.replace("accouch_", "accouch. ").replace("_", " "))
    return f"{human(row)} · {col}"


out = {}
for t, tpl in T.items():
    fields = []
    for key, z in tpl["zones"].items():
        if key.startswith("inline.") and key[7:].split("_2")[0] in IDENTIFIER_LABELS:
            continue                      # identifiers are never asked, never stored
        fields.append(dict(key=key, type=z["type"], label=label(key, z),
                           manual=key.startswith("inline.") or key.startswith("cb.")))
    out[t] = dict(name_fr=NAMES[int(t)][0], name_en=NAMES[int(t)][1], fields=fields)
p = W / "strat15" / "app" / "schema.json"
p.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
print(p, {t: len(v["fields"]) for t, v in out.items()})
