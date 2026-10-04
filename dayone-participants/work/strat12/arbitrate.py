"""Strategy 12: field-level arbitration between reading engines.

Engines: CRNN (with constrained CTC rescoring), a second CRNN checkpoint / model, the VLM (strategy 3).
Rules:
  1. agreement of >= 2 engines on the canonical value -> accept, confidence boosted
  2. otherwise the CRNN's own CTC likelihood arbitrates: each foreign candidate is rescored with the CRNN
     log-probs (a VLM reading the CRNN finds plausible wins over a CRNN reading it finds implausible);
  3. blank guard: if the CRNN says blank with high confidence and only the VLM sees text -> blank (VLM
     hallucination guard), unless new-ink evidence is strong;
  4. remaining disagreement -> À_RÉVISER with all candidates offered as quick replies.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

W = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(W / "strat5")]
from vocab import canonical  # noqa: E402


def arbitrate(cands: dict, rec=None, lp=None, ink: float = 0.0, delta: float = 3.0, crnn_key="crnn"):
    """cands: {engine: value or None}. Returns (value, agreement, alternatives)."""
    canon = {e: canonical(v) for e, v in cands.items()}
    votes = Counter(c for c in canon.values())
    best, n = votes.most_common(1)[0]
    if n >= 2:
        val = next(v for e, v in cands.items() if canon[e] == best)
        return val, n, [v for e, v in cands.items() if canon[e] != best and v]
    base = cands.get(crnn_key)
    if rec is None or lp is None:
        return base, 1, [v for v in cands.values() if v and v != base]
    # blank guard
    if base is None and ink < 0.004:
        return None, 1, [v for v in cands.values() if v]
    scores = {}
    for e, v in cands.items():
        scores[e] = rec.score(lp, v or "")
    e_best = max(scores, key=scores.get)
    if e_best != crnn_key and scores[e_best] < scores[crnn_key] + delta:
        e_best = crnn_key          # a foreign reading must be clearly more likely to override
    val = cands[e_best]
    return val, 1, [v for v in cands.values() if v and v != val]
