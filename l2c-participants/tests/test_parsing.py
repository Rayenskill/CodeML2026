"""Callout grammar, labels, levels and types.  All inputs are hand-written strings."""

import pytest

from l2c_rebar.parsing.elements import find_labels, find_levels, find_sheet_ids, label_candidates, type_from_sheet, type_from_text
from l2c_rebar.parsing.rebar import find_bars, normalize_ocr


def one(text, **kw):
    bars = find_bars(text, **kw)
    assert len(bars) == 1, bars
    return bars[0]


@pytest.mark.parametrize(
    "text, qty, dia, esp, length",
    [
        ("8-25M", 8, "25M", None, None),
        ("VERT.: 8-25M", 8, "25M", None, None),
        ("8 - 25M", 8, "25M", None, None),
        ("(8) 25M VERT.", 8, "25M", None, None),
        ("12(6)-15M", 12, "15M", None, None),
        ("12 (6) -15M", 12, "15M", None, None),
        ("LIG.: 10M@300 c/c", None, "10M", 300, None),
        ("20M@300mm c/c", None, "20M", 300, None),
        ("15M A 300 C/C", None, "15M", 300, None),
        ("ARM. 15M 300 c/c", None, "15M", 300, None),
        ('LIG.: 10M@12" c/c', None, "10M", 305, None),
        ("10M @ 8'' c/c", None, "10M", 203, None),
        ('15M @ 7 1/2"', None, "15M", 190, None),
        ("8-20M x 3000 SUP.", 8, "20M", None, 3000),
        ("8-25M LG 3600", 8, "25M", None, 3600),
        ("5-20M2400 @ 250", 5, "20M", 250, 2400),
        ("COL. 400 x 600 8-25M", 8, "25M", None, None),
    ],
)
def test_plan_and_metric_forms(text, qty, dia, esp, length):
    bar = one(text)
    assert (bar.quantite, bar.diametre, bar.espacement_mm, bar.longueur_mm) == (qty, dia, esp, length)


@pytest.mark.parametrize(
    "text, qty, esp, length, mark",
    [
        ('T: 24 15M 15A12 @12"', 24, 305, None, "15A12"),
        ("B: 24 15M 20-6", 24, None, 6248, None),  # 20 ft 6 in
        ("8 25M 3600", 8, None, 3600, None),
        ("6 20M 20A01B2", 6, None, None, "20A01B2"),
        ("12-15M B12 @ 300 INF.", 12, 300, None, "B12"),
    ],
)
def test_shop_forms(text, qty, esp, length, mark):
    bar = one(text)
    assert (bar.quantite, bar.espacement_mm, bar.longueur_mm, bar.repere) == (qty, esp, length, mark)
    assert bar.conf >= 0.9


def test_roles():
    assert one("LIG.: 10M@300 c/c").role == "ligature"
    assert one("VERT.: 8-25M").role == "vert"
    assert one("T: 24 15M 20-6").role == "sup"
    assert one("12-15M B12 @ 300 INF.").role == "inf"


def test_sub_quantity_is_kept_apart():
    bar = one("12(6)-15M")
    assert bar.quantite == 12 and bar.sub_qty == 6


@pytest.mark.parametrize("text", ["BETON 25MPa", "10MM", "115M", "35MPa / C-1", "3.25M"])
def test_not_a_bar(text):
    assert find_bars(text) == []


def test_bare_size_is_not_informative():
    assert not one("15M").informative


def test_number_before_size_is_not_always_a_quantity():
    assert one("NIV 2 15M @ 300").quantite is None
    a, b = find_bars("15M @ 300 20M @ 250")
    assert (a.espacement_mm, b.quantite, b.espacement_mm) == (300, None, 250)


def test_two_callouts_on_one_line():
    a, b = find_bars("4-25M + 2-20M INF")
    assert (a.quantite, a.diametre, b.quantite, b.diametre) == (4, "25M", 2, "20M")


def test_ocr_confusions():
    assert normalize_ocr("8-2OM") == "8-20M"
    bars = find_bars("8-2OM e 300  1SMe250 I5M@200 2OMPa", ocr=True)
    assert [(b.quantite, b.diametre, b.espacement_mm) for b in bars] == [(8, "20M", 300), (None, "15M", 250), (None, "15M", 200)]


def test_labels():
    assert [l.label for l in find_labels("COLONNE C-12 ET C13A VOIR S-501 TYPE 1 P4", "colonne")] == ["C12", "C13A"]
    assert find_labels("BETON: 35MPa / C-1", "colonne") == []  # exposure class, not a column
    assert [l.label for l in find_labels("SEMELLE S3", "fondation")] == ["S3"]
    assert [l.label for l in find_labels("Q7", "colonne", extra={"Q7"})] == ["Q7"]  # learnt from both documents
    assert label_candidates("C-12 MR3 NIV 2 S-501 15M") == ["C12", "MR3"]


def test_sheets_and_types():
    assert [s for _, _, s in find_sheet_ids("VOIR S-501 ET S-602A")] == ["S-501", "S-602A"]
    assert [type_from_sheet(s) for s in ("S-101", "S-301", "S-401", "S-501", "S-601", "S-001")] == [
        "fondation", "poutre", "mur de refend", "colonne", "dalle", None]
    assert type_from_text("Murs cisaillements") == "mur de refend"
    assert type_from_text("Semelles et radiers") == "fondation"
    assert type_from_text("CONCENTRATION COLONNES") == "mur de refend"


@pytest.mark.parametrize(
    "name, levels",
    [
        ("X_COLONNE-NIV-2@3", ("2", "3")),
        ("X_COLONNE-NIV-FDN@RDC", ("FDN", "RDC")),
        ("X_COLONNE-NIV-11@Toit", ("11", "TOIT")),
        ("X_COLONNE-NIV-SS1@RDC", ("SS1", "RDC")),
        ("X_DALLE TRÉFOND", ("TREFOND",)),
        ("X_SEMELLES FND", ("FDN",)),
        ("X_DALLE NIV2 P1", ("2",)),
        ("X_RADIERS 1@2", ()),
        ("DALLE NIVEAU 3 - ARMATURE SUPÉRIEURE", ("3",)),
    ],
)
def test_levels(name, levels):
    assert find_levels(name) == levels


def test_ocr_pieces_of_one_row_are_read_as_one_line():
    from l2c_rebar.models import TextLine
    from l2c_rebar.pdf.ocr import join_row_fragments

    callout = TextLine("24 10M 10A12", (100, 100, 160, 108), origin="ocr")
    spaces = TextLine("23@300", (168, 100.5, 196, 108.5), origin="ocr")  # the detector cut the line here
    other = TextLine("8 25M 301", (400, 100, 450, 108), origin="ocr")  # same row, another column of the sheet
    below = TextLine("B-12", (100, 130, 120, 138), origin="ocr")
    lines = join_row_fragments([callout, spaces, other, below])
    assert sorted(l.text for l in lines) == ["24 10M 10A12 23@300", "8 25M 301", "B-12"]
    joined = next(l for l in lines if l.text.startswith("24"))
    assert joined.bbox == (100, 100, 196, 108.5)
    assert one(joined.text).espacement_mm == 300


def test_a_spacing_written_on_the_next_line_belongs_to_the_callout_above():
    from l2c_rebar.extract.document import _lacks_spacing
    from l2c_rebar.extract.layout import attach_spacing_lines
    from l2c_rebar.models import TextLine

    callout = TextLine("24 15M 12-06", (100, 100, 160, 108))
    spacing = TextLine('@12"', (110, 110, 130, 118))  # stacked right under it
    elsewhere = TextLine("8 20M 3600", (300, 100, 350, 108))
    # text running up the sheet: the line before is on the left
    up = TextLine("12 10M 10A01", (500, 300, 508, 360), dx=0.0, dy=-1.0)
    up_spacing = TextLine("11@300", (510, 310, 518, 340), dx=0.0, dy=-1.0)
    lines = attach_spacing_lines([callout, spacing, elsewhere, up, up_spacing], _lacks_spacing)
    texts = [l.text for l in lines]
    assert '24 15M 12-06 @12"' in texts and "8 20M 3600" in texts and "12 10M 10A01 11@300" in texts
    assert one('24 15M 12-06 @12"').espacement_mm == 305
    assert one("12 10M 10A01 11@300").espacement_mm == 300
