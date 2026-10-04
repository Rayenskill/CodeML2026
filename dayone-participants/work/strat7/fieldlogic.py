"""Strategy 7 (part 1) + strategy 11 §4.5: per-field grammar, candidate generation and CTC rescoring.

Given the recogniser's log-probs for a zone, we compare the free (greedy) reading with constrained
candidates (field vocabulary in FR/EN/AR, date / BP / number+unit formats).  A candidate wins if its
CTC likelihood is within `delta` nats of the free reading: this repairs missing glyphs ("Ferm" ->
"Fermé"), unit misreads ("3626 9" -> "3626 g") and separator confusions, without inventing values.
"""
from __future__ import annotations

import re
import sys
from functools import lru_cache
from pathlib import Path

W = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(W / "shared"), str(W / "strat5")]
from vocab import (ENUMS, OFFICIAL_PROVINCES, OFFICIAL_REGIONS, UNITS, VARIANT_TO_CODE, infer_kinds,  # noqa: E402
                   to_ascii_digits)

SPECIAL = ["–", "?", "inconnu", "Inconnu", "NSP", "unknown", "غير معروف"]
UNKNOWN_TOKENS = {"?", "inconnu", "nsp", "unknown", "غير معروف", "ne sait pas", "inconnue"}


EXCLUDE_PATIENT = None      # evaluation only: build vocabularies without the patient being read
GRAMMAR_BEAM = True         # grammar-constrained beam search when the free reading is not a valid value
GRAMMAR_KINDS = {"date", "date_or_year", "year", "bp", "int", "numunit", "pattern"}


@lru_cache(maxsize=None)
def _kinds_for(excl):
    return infer_kinds(excl)


def _kinds():
    return _kinds_for(EXCLUDE_PATIENT)


def kind(t: int, key: str) -> dict:
    kinds, _ = _kinds()
    if t == 3 and key.startswith("visites."):
        return kinds.get((t, "visites." + key.split(".")[1]), dict(kind="free"))
    return kinds.get((t, key), dict(kind="free"))


def vocab_for(t: int, key: str) -> tuple:
    return _vocab_for(t, key, EXCLUDE_PATIENT)


@lru_cache(maxsize=None)
def _vocab_for(t: int, key: str, excl) -> tuple:
    """Enumerated candidates for a field (all languages)."""
    k = kind(t, key)
    vals = set()
    if key == "inline.region":
        vals.update(OFFICIAL_REGIONS)
    if key == "inline.province":
        vals.update(OFFICIAL_PROVINCES)
    if k["kind"] == "enum_codes":
        for c in k["codes"]:
            vals.update(ENUMS[c])
    elif k["kind"] == "enum":
        for v in k["values"]:
            vals.add(v)
            code = VARIANT_TO_CODE.get(v.lower())
            if code:
                vals.update(ENUMS[code])
    return tuple(sorted(vals))


def numeric_candidates(text: str, spec: dict) -> list[str]:
    """Format-repaired variants of a numeric reading."""
    t = to_ascii_digits(text)
    digits = re.sub(r"\D", "", t)
    out = []
    k = spec["kind"]
    if k in ("date", "date_or_year") and len(digits) in (8, 6):
        if len(digits) == 8:
            out.append(f"{digits[:2]}/{digits[2:4]}/{digits[4:]}")
        else:
            out.append(f"{digits[:2]}/{digits[2:4]}/20{digits[4:]}")
    if k in ("year", "date_or_year") and len(digits) == 4:
        out.append(digits)
    if k == "bp" and 4 <= len(digits) <= 6:
        for cut in (2, 3):
            if len(digits) - cut in (2, 3):
                out.append(f"{digits[:cut]}/{digits[cut:]}")
    if k == "numunit":
        m = re.search(r"(\d+)(?:[.,](\d+))?", t)
        if m:
            num = m.group(1) + (("." + m.group(2)) if m.group(2) else "")
            nums = {num}
            dec = spec.get("dec", 0)
            if dec and "." not in num and len(num) > 1:
                nums.add(num[:-dec] + "." + num[-dec:])
            unit = spec.get("unit") or ""
            from vocab import UNIT_TO_CANON
            cu = UNIT_TO_CANON.get(unit.lower(), unit)
            uvars = UNITS.get(cu, [unit]) if unit else [""]
            for n in nums:
                for u in set(uvars):
                    if not u:
                        out.append(n)
                    elif u == "k":
                        out.append(f"{n}k")
                    else:
                        out.append(f"{n} {u}")
                out.append(n)
    if k == "pattern":
        tpl = spec["template"]
        nd = sum(c.isdigit() for c in tpl)
        if len(digits) == nd:
            it = iter(digits)
            out.append("".join(next(it) if c.isdigit() else c for c in tpl))
    if k == "int" and digits:
        out.append(digits)
    return out


def grammar_candidates(rec, lps: list, spec: dict, greedy: str) -> set:
    """Best grammatical, plausible strings in the CTC lattices of the views, only when the free reading is not
    one already (blank, '?' or '–' readings are left alone: no value is invented)."""
    from grammar_decode import beam_candidates, grammatical
    if not GRAMMAR_BEAM or spec["kind"] not in GRAMMAR_KINDS or not greedy or greedy in SPECIAL:
        return set()
    if greedy.lower() in UNKNOWN_TOKENS or grammatical(to_ascii_digits(greedy), spec):
        return set()
    out = set()
    for lp in lps:
        out.update(c for _, c in beam_candidates(lp, rec.codec, spec))
    return out


def decide(rec, lp, greedy: str, t: int, key: str, delta: float = 2.5, delta_special: float = 1.0):
    """Return (value, info) after constrained rescoring.

    The best *alternative* candidate (field vocabulary / format repair) replaces the free reading when its
    CTC log-likelihood is within `delta` nats of the free reading's; special tokens ("–", "?", "inconnu")
    need to be within the smaller `delta_special`.
    """
    spec = kind(t, key)
    g_score = rec.score(lp, greedy)
    vocab = set(vocab_for(t, key))
    if greedy:
        vocab.update(numeric_candidates(greedy, spec))
        vocab.update(grammar_candidates(rec, [lp], spec, greedy))
    vocab.discard(greedy)
    special = set(SPECIAL) - {greedy}
    scored = [(rec.score(lp, c), c, delta) for c in vocab] + [(rec.score(lp, c), c, delta_special) for c in special]
    scored.sort(reverse=True)
    empty_s = rec.score(lp, "") if greedy else g_score
    info = dict(greedy=greedy, greedy_score=g_score, best_score=g_score, empty_score=empty_s, snapped=False,
                margin=(g_score - scored[0][0]) if scored else 20.0)
    for s_c, c, d in scored:
        if s_c >= g_score - d:
            info.update(snapped=True, best_score=s_c, margin=s_c - (scored[1][0] if len(scored) > 1 else -1e9))
            return c, info
        break
    return greedy, info


def valid_value(value: str, spec: dict) -> bool:
    """Does a reading respect its field's grammar and plausible range? (always True for free text / enums)"""
    from grammar_decode import grammatical
    from vocab import canonical
    if spec["kind"] not in GRAMMAR_KINDS or not value:
        return True
    v = to_ascii_digits(value).replace("،", ",")
    c = canonical(value) or ""
    if grammatical(v, spec) or grammatical(c, spec) or grammatical(c.replace(" ", ""), spec):
        return True
    # a unit written in a column whose header already gives it ("64,8 kg" under "Poids (kg)")
    return spec["kind"] == "numunit" and not spec.get("unit") and grammatical(c.split(" ")[0], spec)


EMPTY_TAU = 0.9             # a blank reading below this calibrated confidence is asked, not declared NON_FOURNI


def status_for(value: str | None, conf: float, tau_high=0.95, tau_low=0.35) -> str:
    if value is None or value.strip() in ("", "-", "–", "—"):
        return "NON_FOURNI"
    if value.strip().lower() in UNKNOWN_TOKENS:
        return "INCONNU"
    if conf < tau_low:
        return "ILLISIBLE"
    if conf < tau_high:
        return "À_RÉVISER"
    return "CONNU"


def decide_multi(rec, lps: list, t: int, key: str, delta: float = 6.0, delta_special: float = 1.0):
    """String-level ensemble over several views of the same crop (e.g. horizontal stretches 1.0/1.2/1.4):
    candidates = every view's free reading + vocabulary + format repairs; score = mean CTC log-likelihood
    over views (comparable across views, unlike frame posteriors of different lengths)."""
    greedies = [rec.decode(lp)[0].strip() for lp in lps]
    spec = kind(t, key)

    def S(c):
        return sum(rec.score(lp, c) for lp in lps) / len(lps)
    free = {g: S(g) for g in set(greedies)}
    greedy = max(free, key=free.get)
    g_score = free[greedy]
    vocab = set(vocab_for(t, key))
    for g in free:
        if g:
            vocab.update(numeric_candidates(g, spec))
    vocab.update(grammar_candidates(rec, lps, spec, greedy))
    vocab -= set(free)
    special = set(SPECIAL) - set(free)
    scored = sorted([(S(c), c, delta) for c in vocab] + [(S(c), c, delta_special) for c in special], reverse=True)
    empty_s = S("")
    info = dict(greedy=greedy, greedy_score=g_score, best_score=g_score, empty_score=empty_s, snapped=False,
                margin=(g_score - scored[0][0]) if scored else 20.0, views_agree=len(set(greedies)) == 1)
    if scored and scored[0][0] >= g_score - scored[0][2]:
        s_c, c, _ = scored[0]
        info.update(snapped=True, best_score=s_c, margin=s_c - (scored[1][0] if len(scored) > 1 else -1e9))
        return c, info
    return greedy, info
