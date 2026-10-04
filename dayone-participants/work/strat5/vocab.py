"""Strategy 5: value vocabularies (FR / EN / AR), per-field value sampler and normalisation.

* `FIELD_KIND[(page_type, key)]` is inferred from the ground truth values (date, bp, number+unit…),
  with explicit rules for cells never filled in the specimen.
* `sample_value(kind, rng, lang)` draws a plausible value as it would be written.
* `canonical(value)` maps any language / digit system to a canonical comparable form (used by the
  evaluation and by the post-processing).
"""
from __future__ import annotations

import collections
import json
import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
from common import GT_DIR  # noqa: E402

AR_DIGITS = "٠١٢٣٤٥٦٧٨٩"
FA_DIGITS = "۰۱۲۳۴۵۶۷۸۹"
DIGIT_MAP = {ord(a): str(i) for i, a in enumerate(AR_DIGITS)} | {ord(a): str(i) for i, a in enumerate(FA_DIGITS)}

# canonical code -> written variants (FR first)
ENUMS = {
    "RAS": ["RAS", "R.A.S", "NAD", "Nothing to report", "لا شيء"],
    "NEG": ["Neg", "Négatif", "Negative", "Neg.", "سلبي"],
    "POS": ["Pos +", "Positif", "Positive", "Pos", "إيجابي"],
    "NORMALES": ["Normales", "Normal", "طبيعية"],
    "NORMAUX": ["Normaux", "Normal", "طبيعي"],
    "NORMAL": ["Normal", "Normale", "Normal", "طبيعي"],
    "NON": ["Non", "No", "لا"],
    "OUI": ["Oui", "Yes", "نعم"],
    "FERME": ["Fermé", "Closed", "مغلق"],
    "CEPHALIQUE": ["Céphalique", "Cephalic", "رأسي"],
    "SIEGE": ["Siège", "Breech", "مقعدي"],
    "IMMUNE": ["Immune", "Immune", "ممنعة"],
    "NON_IMMUNE": ["Non immune", "Not immune", "غير ممنعة"],
    "AUCUN": ["Aucun", "None", "لا يوجد"],
    "AUCUNE": ["Aucune", "None", "لا يوجد"],
    "NEANT": ["Néant", "None", "لا شيء"],
    "VOIE_BASSE": ["Voie basse", "Vaginal", "ولادة طبيعية"],
    "CESARIENNE": ["Césarienne", "Caesarean", "قيصرية"],
    "MATERNITE": ["Maternité", "Maternity", "دار الولادة"],
    "DOMICILE": ["Domicile", "Home", "المنزل"],
    "PALES": ["Pâles", "Pale", "شاحبة"],
    "PROPRE": ["Propre", "Clean", "نظيفة"],
    "PROPRE_SECHE": ["Propre, sèche", "Clean, dry", "نظيفة وجافة"],
    "CYCLES_REGULIERS": ["Cycles réguliers", "Regular cycles", "دورة منتظمة"],
    "MERE": ["Mère", "Mother", "الأم"],
    "PERE": ["Père", "Father", "الأب"],
    "ONCLE": ["Oncle", "Uncle", "العم"],
    "TANTE": ["Tante", "Aunt", "العمة"],
    "NON_FAIT": ["Non fait", "Not done", "لم يجر"],
    "LYCEE": ["Lycée", "High school", "ثانوي"],
    "COLLEGE": ["Collège", "Middle school", "إعدادي"],
    "SUPERIEUR": ["Supérieur", "University", "جامعي"],
    "PRIMAIRE": ["Primaire", "Primary", "ابتدائي"],
    "F": ["F", "Fille", "Female", "أنثى"],
    "M": ["M", "Garçon", "Male", "ذكر"],
    "ASTHME_LEGER": ["Asthme léger", "Mild asthma", "ربو خفيف"],
    "APPENDICECTOMIE": ["Appendicectomie", "Appendectomy", "استئصال الزائدة"],
    "SFA": ["SFA", "Fetal distress", "ضائقة جنينية"],
    "SOUFFRANCE_FOETALE": ["Souffrance fœtale", "Fetal distress", "ضائقة جنينية"],
    "UTERUS_CICATRICIEL": ["Utérus cicatriciel", "Scarred uterus", "رحم مندب"],
    "PREECLAMPSIE_SEVERE": ["Pré-éclampsie sévère", "Severe pre-eclampsia", "تسمم حمل شديد"],
    "POURSUIVRE_AE": ["Poursuivre l'allaitement exclusif", "Continue exclusive breastfeeding", "متابعة الرضاعة الطبيعية الحصرية"],
    "DISCUTER_MARI": ["Souhaite en discuter avec son mari", "Wants to discuss with husband", "تريد مناقشة الأمر مع زوجها"],
}
UNITS = {  # canonical unit -> variants
    "g": ["g", "g", "غ"], "cm": ["cm", "cm", "سم"], "kg": ["kg", "kg", "كغ"], "SA": ["SA", "wks", "أسبوع"],
    "g/dL": ["g/dL", "g/dL", "غ/دل"], "g/L": ["g/L", "g/L", "غ/ل"], "°C": ["°C", "°C", "°C"],
    "jours": ["jours", "days", "أيام"], "k": ["k", "k", "k"],
}
VARIANT_TO_CODE = {}
for code, vs in ENUMS.items():
    for v in vs:
        VARIANT_TO_CODE.setdefault(v.lower(), code)
UNIT_TO_CANON = {}
for cu, vs in UNITS.items():
    for v in vs:
        UNIT_TO_CANON.setdefault(v.lower(), cu)

# Official administrative divisions of Morocco (public knowledge, used as field vocabularies)
OFFICIAL_REGIONS = ["Tanger-Tétouan-Al Hoceïma", "L'Oriental", "Oriental", "Fès-Meknès", "Rabat-Salé-Kénitra",
                    "Béni Mellal-Khénifra", "Casablanca-Settat", "Marrakech-Safi", "Drâa-Tafilalet", "Souss-Massa",
                    "Guelmim-Oued Noun", "Laâyoune-Sakia El Hamra", "Dakhla-Oued Ed-Dahab"]
OFFICIAL_PROVINCES = [
    "Tanger-Assilah", "M'diq-Fnideq", "Tétouan", "Fahs-Anjra", "Larache", "Al Hoceïma", "Chefchaouen", "Ouezzane",
    "Oujda-Angad", "Nador", "Driouch", "Jerada", "Berkane", "Taourirt", "Guercif", "Figuig",
    "Fès", "Meknès", "El Hajeb", "Ifrane", "Moulay Yacoub", "Sefrou", "Boulemane", "Taounate", "Taza",
    "Rabat", "Salé", "Skhirate-Témara", "Kénitra", "Khémisset", "Sidi Kacem", "Sidi Slimane",
    "Béni Mellal", "Azilal", "Fquih Ben Salah", "Khénifra", "Khouribga",
    "Casablanca", "Mohammedia", "El Jadida", "Nouaceur", "Médiouna", "Benslimane", "Berrechid", "Settat", "Sidi Bennour",
    "Marrakech", "Chichaoua", "Al Haouz", "El Kelâa des Sraghna", "Essaouira", "Rehamna", "Safi", "Youssoufia",
    "Errachidia", "Ouarzazate", "Midelt", "Tinghir", "Zagora",
    "Agadir Ida-Outanane", "Inezgane-Aït Melloul", "Chtouka-Aït Baha", "Taroudant", "Tiznit", "Tata",
    "Guelmim", "Assa-Zag", "Tan-Tan", "Sidi Ifni", "Laâyoune", "Boujdour", "Tarfaya", "Es-Semara",
    "Oued Ed-Dahab", "Aousserd"]
PLACES = ["Maternité", "CHU", "Hôpital provincial", "Domicile", "CSU", "Clinique privée", "Maison d'accouchement"]
EXAMINERS = ["Dr Benjelloun", "Dr Chakir", "Inf. Zahra", "Sage-femme", "Sage-femme Salima", "Sage-femme Hajar",
             "Dr Alaoui", "Inf. Karima", "Dr Idrissi", "SF Naima", "Dr Bennani", "Inf. Fatima"]
EXAMINERS_AR = ["د. بنجلون", "الممرضة زهرة", "القابلة سليمة", "د. الشكير"]
REGIONS = ["Rabat-Salé-Kénitra", "Fès-Meknès", "Marrakech-Safi", "Béni Mellal-Khénifra", "Souss-Massa",
           "Tanger-Tétouan-Al Hoceïma", "Casablanca-Settat", "Oriental", "Drâa-Tafilalet", "Guelmim-Oued Noun"]
PROVINCES = ["Kénitra", "Meknès", "Al Haouz", "Azilal", "Taroudant", "Chefchaouen", "Ifrane", "Errachidia",
             "Settat", "Berkane", "Tiznit", "Larache", "Ouarzazate", "Khémisset", "Sefrou"]
FACILITIES = ["CSCA Al Wifaq", "CSC Hay Salam", "DR Tahannaout Sud", "CSCA Ait Mhamed", "CSC Al Amal",
              "DR Bni Ahmed", "CSU Al Massira", "CSC Ennour", "DR Oulad Ali", "CSUA Al Qods"]
PROFESSIONS = ["Agricultrice", "Commerçante", "Femme au foyer", "Étudiante", "Couturière", "Employée",
               "Ouvrier", "Fonctionnaire", "Mécanicien", "Chauffeur", "Maçon", "Enseignante", "Infirmière",
               "Housewife", "Farmer", "Teacher", "ربة بيت", "فلاحة", "معلمة"]
FREE_TEXT = ["Fer 2 cp/j", "Vit. D 400 UI/j", "Paracétamol", "Acide folique", "Amoxicilline 1g x2",
             "Cicatrice propre", "Contrôle dans 7 jours", "Revoir à J15", "Allaitement mixte", "Sein droit douloureux",
             "Iron 2 tabs/day", "Follow-up in 1 week", "حديد قرصين يوميا", "مراجعة بعد أسبوع"]


def to_ascii_digits(s: str) -> str:
    return s.translate(DIGIT_MAP)


def strip_accents(s: str) -> str:
    s = s.replace("œ", "oe").replace("Œ", "Oe")
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def canonical(value: str | None) -> str | None:
    """Language/format-independent canonical form for comparing extracted vs reference values."""
    if value is None:
        return None
    v = unicodedata.normalize("NFKC", str(value)).strip()
    v = to_ascii_digits(v).replace("،", ",").replace("٫", ".")
    v = re.sub(r"\s+", " ", v)
    if v in {"", "-", "–", "—"}:
        return None
    low = v.lower()
    if low in VARIANT_TO_CODE:
        return VARIANT_TO_CODE[low]
    # dates d/m/y with any separator (before number + unit, which would read "28/09/2025" as 28 + "/09/2025")
    m = re.fullmatch(r"(\d{1,2})\s*[/.\-]\s*(\d{1,2})\s*[/.\-]\s*(\d{2,4})", v)
    if m:
        d, mo, y = m.groups()
        y = ("20" + y) if len(y) == 2 else y
        return f"{int(d):02d}/{int(mo):02d}/{y}"
    m = re.fullmatch(r"(\d{2,3})\s*/\s*(\d{2,3})", v)          # blood pressure
    if m:
        return f"{m.group(1)}/{m.group(2)}"
    # number + unit
    m = re.fullmatch(r"([0-9]+(?:[.,][0-9]+)?)\s*([^\d\s].*)?", v)
    if m:
        num = m.group(1).replace(",", ".")
        unit = (m.group(2) or "").strip()
        cu = UNIT_TO_CANON.get(unit.lower(), unit)
        return f"{num} {cu}".strip()
    return strip_accents(low).replace("’", "'")


# ------------------------------------------------------------------ field kinds
DATE_RE = re.compile(r"^\d{2}/\d{2}/\d{4}$")


def infer_kinds(exclude_patient=None):
    """exclude_patient: leave-one-patient-out vocabularies for honest evaluation on the specimen."""
    vals = collections.defaultdict(list)
    for f in sorted(GT_DIR.glob("page_*.json")):
        g = json.loads(f.read_text(encoding="utf-8"))
        if exclude_patient is not None and g["patient"] == exclude_patient:
            continue
        for x in g["fields"]:
            if x["type"] != "text" or x.get("identifier"):
                continue
            key = x["key"]
            if g["page_type"] == 3 and key.startswith("visites."):
                key = "visites." + key.split(".")[1]
            vals[(g["page_type"], key)].append(x["value"])
    kinds = {}
    for k, vs in vals.items():
        obs = [v for v in vs if v]
        kinds[k] = classify(k, obs)
    return kinds, vals


def classify(k, obs):
    key = k[1]
    if not obs:
        if key.endswith(".nombre"):
            return dict(kind="int", lo=1, hi=4)
        if key.endswith(".date"):
            return dict(kind="date_or_year")
        if ".date." in key:
            return dict(kind="date")
        if key.endswith(".lieu"):
            return dict(kind="enum", values=PLACES)
        if key.endswith("age_gestationnel_sa"):
            return dict(kind="numunit", lo=8, hi=40, dec=0, unit="SA")
        if "poids_nouveau" in key:
            return dict(kind="numunit", lo=1800, hi=4500, dec=0, unit="g")
        if "modalite" in key:
            return dict(kind="enum_codes", codes=["VOIE_BASSE", "CESARIENNE"])
        if "cesarienne" in key:
            return dict(kind="enum_codes", codes=["SFA", "UTERUS_CICATRICIEL", "SOUFFRANCE_FOETALE"])
        if "compl" in key:
            return dict(kind="enum_codes", codes=["RAS"])
        return dict(kind="free")
    if all(re.fullmatch(r"\d+(-\d+)+", v) for v in obs):
        return dict(kind="pattern", template=obs[0])
    if all(DATE_RE.match(v) for v in obs):
        return dict(kind="date")
    if all(re.fullmatch(r"\d{4}", v) for v in obs):
        return dict(kind="year")
    if all(re.fullmatch(r"\d{2,3}/\d{2,3}", v) for v in obs):
        return dict(kind="bp")
    m = [re.fullmatch(r"(\d+(?:\.\d+)?)\s*(.*)", v) for v in obs]
    if all(m):
        nums = [float(x.group(1)) for x in m]
        units = collections.Counter(x.group(2) for x in m)
        dec = max((len(x.group(1).split(".")[1]) if "." in x.group(1) else 0) for x in m)
        return dict(kind="numunit", lo=min(nums), hi=max(nums), dec=dec, unit=units.most_common(1)[0][0])
    codes = [VARIANT_TO_CODE.get(v.lower()) for v in obs]
    if all(codes):
        return dict(kind="enum_codes", codes=sorted(set(codes)))
    return dict(kind="enum", values=sorted(set(obs)))


def _digits(s, rng, lang):
    if lang == "ar" and rng.random() < 0.5:
        return "".join(AR_DIGITS[int(c)] if c.isdigit() else c for c in s)
    return s


def sample_value(spec, rng, lang="fr", key=""):
    """Value as written on paper. lang in {'fr','en','ar'}."""
    li = {"fr": 0, "en": 1, "ar": 2}[lang]
    k = spec["kind"]
    if k in ("date", "date_or_year"):
        if k == "date_or_year" and rng.random() < 0.5:
            return _digits(str(rng.integers(2010, 2027)), rng, lang)
        d, m, y = rng.integers(1, 29), rng.integers(1, 13), rng.integers(2015, 2028)
        sep = "/" if rng.random() < 0.9 else rng.choice(["-", "."])
        return _digits(f"{d:02d}{sep}{m:02d}{sep}{y}", rng, lang)
    if k == "pattern":
        return _digits("".join(str(rng.integers(0, 10)) if c.isdigit() else c for c in spec["template"]), rng, lang)
    if k == "year":
        return _digits(str(rng.integers(2010, 2027)), rng, lang)
    if k == "bp":
        return _digits(f"{rng.integers(85, 150)}/{rng.integers(45, 95)}", rng, lang)
    if k == "int":
        return _digits(str(rng.integers(spec["lo"], spec["hi"] + 1)), rng, lang)
    if k == "numunit":
        lo, hi = spec["lo"], spec["hi"]
        span = max(hi - lo, 1)
        x = rng.uniform(lo - 0.3 * span, hi + 0.3 * span)
        x = max(x, 0)
        num = f"{x:.{spec['dec']}f}"
        unit = spec["unit"]
        if unit:
            cu = UNIT_TO_CANON.get(unit.lower(), unit)
            uv = UNITS.get(cu, [unit, unit, unit])[li]
            sep = "" if unit == "k" else " "
            return _digits(f"{num}{sep}{uv}", rng, lang)
        return _digits(num, rng, lang)
    if k == "enum_codes":
        code = rng.choice(spec["codes"]) if rng.random() < 0.85 else rng.choice(list(ENUMS))
        vs = ENUMS[code]
        if lang == "fr":
            return vs[0]
        if lang == "en":
            return vs[1] if len(vs) > 2 else vs[0]
        return vs[-1]
    if k == "enum":
        v = str(rng.choice(spec["values"]))
        code = VARIANT_TO_CODE.get(v.lower())
        if code and lang != "fr":
            vs = ENUMS[code]
            return vs[1] if lang == "en" else vs[-1]
        if "examen_fait_par" in key or "vu_par" in key:
            return str(rng.choice(EXAMINERS_AR)) if lang == "ar" else str(rng.choice(EXAMINERS))
        return v
    # free text
    return str(rng.choice(FREE_TEXT))


if __name__ == "__main__":
    kinds, vals = infer_kinds()
    import numpy as np
    rng = np.random.default_rng(0)
    for k, s in sorted(kinds.items()):
        print(k, s["kind"], [sample_value(s, rng, l, k[1]) for l in ("fr", "en", "ar")])
