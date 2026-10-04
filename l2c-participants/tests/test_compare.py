"""Bar reconciliation and page layout reasoning, on hand-built objects."""

from l2c_rebar.compare import compare_bars
from l2c_rebar.config import Config
from l2c_rebar.extract.layout import Label, assign_labels, cluster_stacked
from l2c_rebar.models import Bar

CFG = Config()


def bar(dia, qty=None, esp=None, length=None, role=None, box=(0, 0, 10, 10)):
    return Bar(diametre=dia, quantite=qty, espacement_mm=esp, longueur_mm=length, role=role, bbox=box)


def test_identical_bars_conform():
    comp = compare_bars([bar("25M", 8), bar("10M", esp=300)], [bar("25M", 8), bar("10M", 14, 300)], CFG)
    assert comp.conform and comp.checked == 2


def test_quantity_shortfall_is_critical():
    comp = compare_bars([bar("25M", 8)], [bar("25M", 6)], CFG)
    (e,) = comp.ecarts
    assert (e.attribut, e.plan, e.atelier, e.gravite) == ("quantite", 8, 6, "critique")


def test_quantity_surplus_is_minor():
    (e,) = compare_bars([bar("25M", 8)], [bar("25M", 10)], CFG).ecarts
    assert e.gravite == "mineur"


def test_wider_spacing_is_critical_and_tighter_is_minor():
    (wide,) = compare_bars([bar("10M", esp=200)], [bar("10M", 20, 250)], CFG).ecarts
    (tight,) = compare_bars([bar("10M", esp=200)], [bar("10M", 20, 150)], CFG).ecarts
    assert (wide.attribut, wide.gravite, tight.gravite) == ("espacement_mm", "critique", "mineur")


def test_smaller_diameter_is_detected():
    (e,) = compare_bars([bar("25M", 8)], [bar("20M", 8)], CFG).ecarts
    assert (e.attribut, e.plan, e.atelier, e.gravite) == ("diametre", "25M", "20M", "critique")


def test_plan_quantity_split_over_two_shop_marks():
    assert compare_bars([bar("25M", 8)], [bar("25M", 4), bar("25M", 4)], CFG).conform


def test_missing_bars_are_reported():
    (e,) = compare_bars([bar("25M", 8)], [bar("10M", 14, 300)], CFG).ecarts
    assert e.attribut == "absence"


def test_nothing_comparable_is_unknown_not_an_error():
    comp = compare_bars([bar("15M", esp=300)], [bar("15M", 24)], CFG)
    assert comp.conform and comp.checked == 0 and comp.unknown == 1


def test_length_tolerance():
    assert compare_bars([bar("20M", 8, length=3000)], [bar("20M", 8, length=3010)], CFG).conform
    (e,) = compare_bars([bar("20M", 8, length=3000)], [bar("20M", 8, length=2700)], CFG).ecarts
    assert e.attribut == "longueur_mm"


def test_header_row_owns_the_cells_below():
    labels = [Label(f"C{i}", f"C{i}", (240 + 150 * i, 110, 260 + 150 * i, 122)) for i in range(4)]
    cells = [bar("25M", 8, box=(235 + 150 * i, 400, 265 + 150 * i, 410)) for i in range(4)]
    assert [l.label for l in assign_labels(cells, labels, 220)] == ["C0", "C1", "C2", "C3"]


def test_first_column_owns_its_row():
    labels = [Label(f"S{i}", f"S{i}", (100, 145 + 40 * i, 112, 155 + 40 * i)) for i in range(4)]
    cells = [bar("20M", 8, box=(380, 145 + 40 * i, 440, 155 + 40 * i)) for i in range(4)]
    assert [l.label for l in assign_labels(cells, labels, 220)] == ["S0", "S1", "S2", "S3"]


def test_nearest_label_for_details_and_none_when_far():
    labels = [Label("MR1", "MR1", (150, 410, 180, 422))]
    near, far = bar("25M", 8, box=(150, 360, 180, 370)), bar("25M", 8, box=(900, 60, 930, 70))
    owner_near, owner_far = assign_labels([near, far], labels, 220)
    assert owner_near.label == "MR1" and owner_far is None


def test_stacked_callouts_form_one_annotation():
    a = bar("25M", 8, box=(100, 100, 140, 109))
    b = bar("10M", esp=300, box=(100, 114, 180, 123))
    c = bar("15M", 5, box=(600, 400, 640, 409))
    groups = sorted(cluster_stacked([a, b, c], 1.4), key=len)
    assert [len(g) for g in groups] == [1, 2]
