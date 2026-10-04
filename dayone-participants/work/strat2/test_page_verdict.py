"""An unrecognised page never yields confident fields (jury risk: real booklets with another layout)."""
import sys
from pathlib import Path

import cv2
import pytest

W = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(W / s) for s in ("shared", "strat2", "strat6", "strat7", "strat11", "strat18")]
from common import MODELS as M  # noqa: E402
pytestmark = pytest.mark.skipif(not (M / "crnn_final.pt").exists(), reason="trained model not available")


@pytest.fixture(scope="module")
def ext():
    from calibrate import Calibrator
    from extract_zonal import Extractor
    from omr import OMR
    from recognizer import Recognizer
    return Extractor(Recognizer(str(M / "crnn_final.pt")), OMR(str(M / "omr_v2.pt")), Calibrator.load(str(M / "calibrator.json")))


def test_real_booklet_photos_are_rejected(ext):
    from common import REG
    for n in range(1, 6):
        p = ext.extract(cv2.cvtColor(cv2.imread(str(REG / f"1-{n}.jpg")), cv2.COLOR_BGR2RGB))
        assert p["page_status"] == "PAGE_NON_RECONNUE", n
        assert not any(f["status"] == "CONNU" for f in p["fields"]), n


def test_specimen_page_accepted(ext):
    from common import png_for_page
    p = ext.extract(cv2.cvtColor(cv2.imread(str(png_for_page(19))), cv2.COLOR_BGR2RGB))
    assert p["page_status"] == "OK" and p["page_type"] == 3
    resolved = sum(f["status"] in ("CONNU", "NON_FOURNI", "NON_APPLICABLE") for f in p["fields"])
    assert resolved / len(p["fields"]) > 0.9          # a clean page needs only a few questions
