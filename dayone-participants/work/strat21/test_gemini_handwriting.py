"""Offline tests of strategy 21: privacy guard, sheet segmentation, verification and bank -> synthetic pages."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gemini_handwriting as gh  # noqa: E402


def test_guard_refuses_unsampled_or_identifier_values():
    g = gh.Guard()
    items = gh.sample_sheet(np.random.default_rng(0), g, 6)
    vals = [x["text"] for x in items]
    g.check_text_request(vals, gh.build_prompt(items, np.random.default_rng(0)))     # sampled values pass
    with pytest.raises(gh.PrivacyError):
        g.check_text_request(vals + ["14/02/2026 Fatima"], "x")                      # not from the sampler
    g.register_value("CB609814")
    with pytest.raises(gh.PrivacyError):
        g.check_text_request(["CB609814"], "x")                                      # CIN-like value
    with pytest.raises(gh.PrivacyError):
        g.check_image_part(b"\x89PNG photo read from disk")                          # never send local images


def test_dry_run_bank_and_synthesis(tmp_path):
    stats = gh.build_bank(3, tmp_path, "dry", dry_run=True, per_sheet=8, seed=1)
    assert stats["kept"] >= 12, stats                    # most lines survive segmentation + verification
    bank = gh.load_bank(tmp_path)
    assert sum(len(v) for v in bank.values()) == stats["kept"]
    import synth_pages as sp
    rng = np.random.default_rng(0)
    used = 0
    for t in (2, 3):
        page, fields, _ = sp.synth_page(t, rng, holdout_specimen=False, bank=bank, bank_p=1.0)
        texts = {e["text"] for v in bank.values() for e in v}
        used += sum(1 for f in fields if f["type"] == "text" and f.get("font", "").endswith(".png"))
        for f in fields:                                  # GT of a bank-drawn field is the bank text
            if f.get("font", "").endswith(".png"):
                assert f["value"] in texts
    assert used > 0


def test_synth_page_unchanged_without_bank():
    import synth_pages as sp
    a = sp.synth_page(4, np.random.default_rng(5), holdout_specimen=False)
    b = sp.synth_page(4, np.random.default_rng(5), holdout_specimen=False, bank=None, bank_p=0.5, hw_aug=0.0)
    assert np.array_equal(a[0], b[0]) and [f["value"] for f in a[1]] == [f["value"] for f in b[1]]
