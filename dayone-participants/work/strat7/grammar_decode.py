"""Strategy 7 / 11: grammar-constrained CTC beam search for formatted fields (dates, BP, numbers + unit, codes).

The free (greedy) reading of a blurred "152" can collapse to "1" or "12/91"; the right digits are usually still
the 2nd/3rd choice of the frames. Instead of only re-formatting the greedy string, we search the CTC lattice for
the most likely strings that *match the field's grammar* and lie in a plausible range, and hand them to the
rescoring of fieldlogic.decide_multi as extra candidates.

Grammars are small automata built with Thompson combinators (lit / cls / seq / alt / rep / opt) and determinised
lazily; the prefix beam search only extends a prefix with characters the automaton accepts.
"""
from __future__ import annotations

import math
import re
from functools import lru_cache

import numpy as np

DIG = "0123456789"
NEG = -1e30


class NFA:
    def __init__(self):
        self.eps, self.tr = [], []          # per state: [states], {char: [states]}

    def new(self):
        self.eps.append([]); self.tr.append({})
        return len(self.eps) - 1


# Thompson fragments: functions nfa -> (start, end)
def lit(s):
    def f(n):
        a = n.new(); cur = a
        for c in s:
            b = n.new(); n.tr[cur].setdefault(c, []).append(b); cur = b
        return a, cur
    return f


def cls(chars):
    def f(n):
        a, b = n.new(), n.new()
        for c in chars:
            n.tr[a].setdefault(c, []).append(b)
        return a, b
    return f


def seq(*parts):
    def f(n):
        s, e = parts[0](n)
        for p in parts[1:]:
            s2, e2 = p(n); n.eps[e].append(s2); e = e2
        return s, e
    return f


def alt(*parts):
    def f(n):
        a, b = n.new(), n.new()
        for p in parts:
            s, e = p(n); n.eps[a].append(s); n.eps[e].append(b)
        return a, b
    return f


def opt(p):
    return alt(p, lit(""))


def rep(p, lo, hi):
    return seq(*([p] * lo + [opt(p)] * (hi - lo))) if hi > 0 else lit("")


class DFA:
    def __init__(self, frag):
        self.n = NFA()
        s, self.end = frag(self.n)
        self.start = self._close({s})
        self._step, self._chars = {}, {}

    def _close(self, states):
        st, seen = list(states), set(states)
        while st:
            x = st.pop()
            for y in self.n.eps[x]:
                if y not in seen:
                    seen.add(y); st.append(y)
        return frozenset(seen)

    def step(self, S, c):
        k = (S, c)
        if k not in self._step:
            nxt = {y for x in S for y in self.n.tr[x].get(c, ())}
            self._step[k] = self._close(nxt) if nxt else None
        return self._step[k]

    def chars(self, S):
        if S not in self._chars:
            self._chars[S] = sorted({c for x in S for c in self.n.tr[x]})
        return self._chars[S]

    def accepts(self, S):
        return self.end in S

    def match(self, s):
        S = self.start
        for c in s:
            S = self.step(S, c)
            if S is None:
                return False
        return self.accepts(S)


# ----------------------------------------------------------------------------------------- field grammars
SEP = cls("/-.")
D = cls(DIG)


def _unit_alts(unit):
    from vocab import UNIT_TO_CANON, UNITS
    cu = UNIT_TO_CANON.get(unit.lower(), unit)
    vs = {v for v in UNITS.get(cu, [unit]) if v and not re.search(r"[؀-ۿ]", v)}   # Latin variants
    return sorted(vs)


def grammar_for(spec: dict):
    k = spec["kind"]
    date = seq(rep(D, 1, 2), SEP, rep(D, 1, 2), SEP, rep(D, 4, 4))
    if k == "date":
        return date
    if k == "date_or_year":
        return alt(date, rep(D, 4, 4))
    if k == "year":
        return rep(D, 4, 4)
    if k == "bp":
        return seq(rep(D, 2, 3), lit("/"), rep(D, 2, 3))
    if k == "int":
        return rep(D, 1, max(1, len(str(int(spec.get("hi", 99)))) + 1))
    if k == "pattern":
        return seq(*[D if c.isdigit() else lit(c) for c in spec["template"]])
    if k == "numunit":
        nd = max(1, len(str(int(spec.get("hi", 999)))))
        num = rep(D, 1, nd + 1)
        if spec.get("dec"):
            num = seq(num, opt(seq(cls(".,"), rep(D, 1, spec["dec"]))))
        unit = spec.get("unit") or ""
        if not unit:
            return num
        us = _unit_alts(unit)
        if unit == "k":
            return seq(num, opt(lit("k")))
        return seq(num, opt(seq(opt(lit(" ")), alt(*[lit(u) for u in us]))))
    return None


@lru_cache(maxsize=None)
def _dfa(spec_key):
    g = grammar_for(dict(spec_key))
    return DFA(g) if g else None


def spec_key(spec):
    return tuple(sorted((k, v) for k, v in spec.items() if isinstance(v, (str, int, float))))


def plausible(s: str, spec: dict) -> bool:
    """Range checks that the grammar alone cannot express."""
    k = spec["kind"]
    nums = [int(x) for x in re.findall(r"\d+", s)]
    if k in ("date", "date_or_year") and len(nums) == 3:
        d, m, y = nums
        return 1 <= d <= 31 and 1 <= m <= 12 and 1990 <= y <= 2035
    if k in ("year", "date_or_year") and len(nums) == 1:
        return 1950 <= nums[0] <= 2035
    if k == "bp" and len(nums) == 2:
        return 60 <= nums[0] <= 250 and 30 <= nums[1] <= 160 and nums[0] > nums[1]
    if k in ("numunit", "int"):
        m = re.match(r"\d+(?:[.,]\d+)?", s)
        if not m:
            return False
        x = float(m.group().replace(",", "."))
        lo, hi = float(spec.get("lo", 0)), float(spec.get("hi", 1e9))
        span = max(hi - lo, 1.0)                     # observed range, widened (relative for narrow ranges)
        return min(lo - span, 0.5 * lo) <= x <= max(hi + span, 1.5 * hi)
    return True


def beam_candidates(lp, codec, spec: dict, beam: int = 12, topk: int = 5, prune: float = -9.0):
    """Most likely strings under CTC that match the field grammar (visual order = logical for these alphabets).
    lp: T x K log-probs (torch or numpy). Returns [(logp, string)] best first (approximate prefix beam search)."""
    dfa = _dfa(spec_key(spec))
    if dfa is None:
        return []
    x = (lp.float().numpy() if hasattr(lp, "numpy") else np.asarray(lp, np.float32)).tolist()
    idx = codec.idx
    beams = {"": (0.0, NEG, dfa.start)}                    # prefix -> (log p ending in blank, non-blank, state)

    def la(a, b):
        if a < b:
            a, b = b, a
        return a if b <= NEG else a + math.log1p(math.exp(b - a))
    for row in x:
        nb = {}

        def add(p, b, n, S):
            ob, on, _ = nb.get(p, (NEG, NEG, S))
            nb[p] = (la(ob, b), la(on, n), S)
        for p, (pb, pn, S) in beams.items():
            tot = la(pb, pn)
            add(p, tot + row[0], NEG, S)                    # blank
            if p:
                last = p[-1]
                add(p, NEG, pn + row[idx[last]], S)         # repeated last char, collapsed
            for c in dfa.chars(S):
                k = idx.get(c)
                if k is None or row[k] < prune:
                    continue
                S2 = dfa.step(S, c)
                ext = (pb if p and p[-1] == c else tot) + row[k]
                add(p + c, NEG, ext, S2)
        beams = dict(sorted(nb.items(), key=lambda kv: -la(kv[1][0], kv[1][1]))[:beam])
    out = [(float(la(pb, pn)), p) for p, (pb, pn, S) in beams.items() if p and dfa.accepts(S) and plausible(p, spec)]
    out.sort(reverse=True)
    return out[:topk]


def grammatical(s: str, spec: dict) -> bool:
    dfa = _dfa(spec_key(spec))
    return bool(dfa and s and dfa.match(s) and plausible(s, spec))
