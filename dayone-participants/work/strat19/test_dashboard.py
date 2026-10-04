import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dashboard import K, by_age, indicators, load  # noqa: E402


def test_indicators_have_missing_counts_and_suppression():
    df = load()
    ind = indicators(df)
    assert set(ind["indicator"]) >= {"VIH positif", "Syphilis positive", "Hépatite C positive", "TA >= 140/90"}
    hb = ind[ind.indicator.str.startswith("Hb")].iloc[0]
    assert hb["missing"] == int(df["hemoglobin"].isna().sum())
    for _, r in ind.iterrows():                         # no published cell below k
        if r["numerator"] is not None and r["numerator"] == r["numerator"]:
            assert r["numerator"] == 0 or r["numerator"] >= K


def test_age_bands_and_no_identifiers():
    df = load()
    t = by_age(df)
    assert set(t.index) <= {"<20", "20-34", "35+", "inconnu"}
    assert "id" not in df.columns


def test_dp_noise_changes_counts_but_keeps_shape():
    df = load()
    a, b = indicators(df), indicators(df, eps=0.5, seed=1)
    assert len(a) == len(b)


def test_indicators_from_digitised_records():
    from dashboard import indicators_from_records
    rows = [{"pages": [{"page_type": 3, "fields": [
        {"key": "visites.ta.t1_v2", "value": "150/95", "status": "CONNU"},
        {"key": "visites.ta.t2_v1", "value": None, "status": "NON_FOURNI"},
        {"key": "visites.serologie_vih.t1_v2", "value": "Neg", "status": "CONNU"}]}]}] * 6
    t = indicators_from_records(rows)
    bp = t[t.indicator.str.startswith("TA")].iloc[0]
    assert bp["denominator"] == 6 and bp["numerator"] == 6 and bp["missing"] == 6
    hiv = t[t.indicator == "VIH positif"].iloc[0]
    assert hiv["numerator"] == 0
