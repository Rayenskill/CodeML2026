"""Grammar automata + constrained CTC beam search (strategy 7/11)."""
import sys
from pathlib import Path

import numpy as np

W = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(W / s) for s in ("shared", "strat5", "strat7", "strat11")]
from grammar_decode import _dfa, beam_candidates, grammatical, spec_key  # noqa: E402

DATE = dict(kind="date")
BCF = dict(kind="numunit", lo=120, hi=160, dec=0, unit="")
POIDS = dict(kind="numunit", lo=55, hi=80, dec=1, unit="kg")
BP = dict(kind="bp")


def test_grammars():
    assert grammatical("12/03/2026", DATE) and grammatical("1-3-2026", DATE)
    assert not grammatical("12/13/2026", DATE) and not grammatical("12/03/26", DATE)
    assert grammatical("141/88", BP) and not grammatical("88/141", BP) and not grammatical("14188", BP)
    assert grammatical("152", BCF) and not grammatical("1", BCF) and not grammatical("12/91", BCF)
    assert grammatical("64.3 kg", POIDS) and grammatical("64.3", POIDS) and grammatical("64,3kg", POIDS)
    assert not grammatical("64.3 cm", POIDS)
    assert _dfa(spec_key(dict(kind="free"))) is None


class Codec:
    def __init__(self, chars):
        self.idx = {c: i + 1 for i, c in enumerate(chars)}


def lattice(frames, codec, K):
    """frames: list of {char or '' (blank): prob} -> T x K log-probs."""
    lp = np.full((len(frames), K), 1e-6)
    for t, fr in enumerate(frames):
        for c, p in fr.items():
            lp[t, 0 if c == "" else codec.idx[c]] = p
    lp /= lp.sum(1, keepdims=True)
    return np.log(lp)


def test_beam_recovers_second_choice_digits():
    codec = Codec("0123456789/.- kgSA")
    K = len(codec.idx) + 1
    # greedy reads "1" then blanks ("152" collapsed by blur); 5 and 2 are second choices
    frames = [{"1": .9, "": .1}, {"": .6, "5": .4}, {"": .55, "2": .45}, {"": 1.0}]
    lp = lattice(frames, codec, K)
    cands = beam_candidates(lp, codec, BCF)
    assert cands and cands[0][1] == "152", cands
    # BP: separator missing in the greedy, recovered from the lattice
    frames = [{"1": 1}, {"4": 1}, {"1": 1}, {"": .7, "/": .3}, {"8": 1}, {"": 1}, {"8": 1}]
    cands = beam_candidates(lattice(frames, codec, K), codec, BP)
    assert cands[0][1] == "141/88", cands
