"""End-to-end runs on the synthetic project (no confidential data involved)."""

import json

import pymupdf
import pytest

from l2c_rebar.cli import main
from l2c_rebar.config import Config
from l2c_rebar.models import AJOUTE, MANQUANT, NON_CONFORME, Record
from l2c_rebar.pdf.text import text_layer_lines
from l2c_rebar.pipeline import discover, run_project, shop_file_hints
from l2c_rebar.report.annotate import annotate_project
from l2c_rebar.report.pdf_report import build_report
from l2c_rebar.revisions import diff_revisions
from l2c_rebar.synthetic import ADDED, MISSING, PLANTED, build_project


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    root = tmp_path_factory.mktemp("demo")
    project = build_project(root)
    return run_project(project, root / "out", Config(ocr="off"))


def test_discovery_reads_type_and_levels_from_names(tmp_path):
    project = build_project(tmp_path)
    files = discover(project)
    assert len(files.plans) == 1 and len(files.shops) == 6
    hints = {p.stem: shop_file_hints(p, files.root) for p in files.shops}
    assert hints["DEMO_COLONNE-NIV-RDC@2"] == ("colonne", ("RDC", "2"))
    assert hints["DEMO_MURS REFENDS"] == ("mur de refend", ())
    assert hints["DEMO_DALLE NIV 2"] == ("dalle", ("2",))


def test_discovery_tolerates_another_folder_layout(tmp_path):
    def pdf(path):
        path.parent.mkdir(parents=True, exist_ok=True)
        doc = pymupdf.open()
        doc.new_page()
        doc.save(path)

    root = tmp_path / "PROJET"
    pdf(root / "Plans" / "STR_S-501.pdf")
    pdf(root / "Plans" / "STR_S-601.pdf")
    pdf(root / "Dessins d'atelier" / "Colonnes" / "COL NIV 2.pdf")
    pdf(root / "Dessins d'atelier" / "Dalles" / "PLAN DALLE NIV 2.pdf")  # "PLAN" in a shop drawing's name
    pdf(root / "Dessins_atelier" / "Murs" / "PLAN MUR NIV 3.pdf")
    files = discover(root)
    assert sorted(p.name for p in files.plans) == ["STR_S-501.pdf", "STR_S-601.pdf"]
    assert sorted(p.name for p in files.shops) == ["COL NIV 2.pdf", "PLAN DALLE NIV 2.pdf", "PLAN MUR NIV 3.pdf"]
    assert shop_file_hints(files.shops[0], root)[0] in ("colonne", "dalle")

    pdf(root / "L2C_PLAN_STR_PROJET.pdf")  # a plan at the root is preferred to plans elsewhere
    assert [p.name for p in discover(root).plans] == ["L2C_PLAN_STR_PROJET.pdf"]


def test_every_planted_discrepancy_is_found(run):
    found = [(r.type_element, e.attribut, e.plan, e.atelier) for r in run.results if r.statut == NON_CONFORME
             for e in r.ecarts]
    for p in PLANTED:
        assert (p["type"], p["attribut"], p["plan"], p["atelier"]) in found, p
    assert len([r for r in run.results if r.statut == NON_CONFORME]) == len(PLANTED)  # and no false alarm


def test_missing_and_added_elements(run):
    missing = [r for r in run.results if r.statut == MANQUANT]
    added = [r for r in run.results if r.statut == AJOUTE]
    assert [(r.element, r.plan.levels) for r in missing] == [(MISSING, ("RDC",))]
    assert [r.atelier.label for r in added] == [ADDED]


def test_storey_selects_the_schedule_row(run):
    c2 = [r for r in run.results if r.plan and r.plan.label == "C2" and r.atelier]
    assert {(r.plan.levels, r.atelier.levels) for r in c2} == {(("RDC",), ("RDC", "2")), (("2",), ("2", "3"))}
    assert all(r.statut == "conforme" for r in c2)


def test_json_follows_appendix_a(run):
    records = json.loads(run.files["elements"].read_text(encoding="utf-8"))
    assert records and len({r["id"] for r in records}) == len(records)
    keys = {"id", "source", "fichier", "feuillet", "page", "x", "y", "type_element", "element", "armature"}
    bar_keys = {"repere", "diametre", "quantite", "espacement_mm", "longueur_mm"}
    for raw in records:
        assert set(raw) == keys
        assert all(set(a) == bar_keys for a in raw["armature"])
        Record.model_validate(raw)
    assert {r["source"] for r in records} == {"plan", "atelier"}
    assert {r["type_element"] for r in records} == {"fondation", "poutre", "mur de refend", "colonne", "dalle"}


def test_coordinates_point_at_the_annotation(run):
    s3 = next(e for e in run.plan_elements if e.label == "S3")
    assert (s3.feuillet, s3.page) == ("S-101", 1)
    assert abs(s3.x - 106) < 8 and abs(s3.y - 227) < 8  # "S3" is written at (100, 230)


def test_report_and_annotated_pdfs(run, tmp_path):
    report = build_report(run.project, run.results, run.coverage, run.stats, tmp_path / "rapport.pdf", Config())
    with pymupdf.open(report) as doc:
        text = "".join(page.get_text() for page in doc)
    for sheet in ("S-101", "S-301", "S-401", "S-501", "S-601"):
        assert sheet in text
    assert "NC-0001" in text and "manque 2" in text
    assert "Liste des écarts" in text and "Quantité 8 -> 6 (25M)" in text  # every finding on one list first

    rows = run.files["ecarts"].read_text(encoding="utf-8-sig").splitlines()
    assert rows[0].startswith("id;statut;feuillet") and len(rows) - 1 == sum(r.statut != "conforme" for r in run.results)

    written = annotate_project(run.results, tmp_path)
    plan = next(p for src, p in written.items() if "PLAN" in src)
    with pymupdf.open(plan) as doc:
        assert sum(len(list(page.annots() or [])) for page in doc) >= 2 * len(PLANTED)
        links = [l for page in doc for l in page.get_links()]
    assert any(l.get("file", "").endswith("_annote.pdf") for l in links)


def test_text_coordinates_follow_page_rotation():
    doc = pymupdf.open()
    page = doc.new_page(width=400, height=200)
    page.insert_text((50, 60), "8-25M", fontsize=12)
    page.set_rotation(90)
    (line,) = text_layer_lines(page)
    x0, y0, x1, y1 = line.bbox
    assert page.rect.width == 200 and 130 < x0 < x1 < 160 and 45 < y0 < y1 < 90
    assert line.vertical


def test_cli_and_revision_diff(tmp_path, capsys):
    project = build_project(tmp_path)
    assert main(["run", str(project), "--out", str(tmp_path / "out"), "--ocr", "off", "--no-crops", "--quiet"]) == 0
    assert (tmp_path / "out" / "DEMO_rapport.pdf").stat().st_size > 2000
    assert "5 non conformes" in capsys.readouterr().out

    old = project / "DA" / "Colonnes" / "DEMO_COLONNE-NIV-RDC@2.pdf"
    new = project / "DA" / "Colonnes" / "DEMO_COLONNE-NIV-2@3.pdf"
    changes = diff_revisions(old, new, Config(ocr="off"))
    assert {c["apres"]["element"] for c in changes["modifies"]} == {"C1", "C2", "C3"}
    assert len(changes["ajoutes"]) == 2 and not changes["retires"]


@pytest.mark.slow
def test_page_without_text_layer_is_read_by_ocr(tmp_path):
    project = build_project(tmp_path, raster_slab=True)
    out = run_project(project, tmp_path / "out", Config(workers=1, ocr_dpi=150))
    assert out.stats["atelier_pages_ocr"] == 1
    slab = [(e.plan, e.atelier) for r in out.results if r.type_element == "dalle" for e in r.ecarts]
    assert (10, 8) in slab


@pytest.mark.parametrize("rotation", [0, 90, 270])
def test_annotation_lands_on_the_text_of_a_rotated_page(tmp_path, rotation):
    import numpy as np

    from l2c_rebar.models import Element, Result
    from l2c_rebar.report.crops import Cropper

    src = tmp_path / "shop.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=400, height=200)
    page.insert_text((50, 60), "8-25M", fontsize=12)
    page.set_rotation(rotation)
    (line,) = text_layer_lines(page)
    doc.save(src)
    doc.close()

    x0, y0, x1, y1 = line.bbox
    el = Element(source="atelier", fichier=src.name, feuillet="shop", page=1, type_element="colonne", element="C1",
                 x=(x0 + x1) / 2, y=(y0 + y1) / 2, path=str(src), label="C1", labeled=True)
    result = Result(statut=AJOUTE, type_element="colonne", element="C1", feuillet="S-501", atelier=el, id="AJ-0001")
    (annotated,) = annotate_project([result], tmp_path).values()

    with pymupdf.open(annotated) as out:
        clip = pymupdf.Rect(x0 - 12, y0 - 12, x1 + 12, y1 + 12)  # displayed coordinates
        pix = out[0].get_pixmap(clip=clip, dpi=144)
    rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3].astype(int)
    blue_ring = (rgb[:, :, 2] > 150) & (rgb[:, :, 0] < 90)
    assert blue_ring.sum() > 50  # the circle is drawn around the text, wherever the page rotation put it

    cropper = Cropper()
    assert cropper.crop(str(src), 1, [line.bbox])[:4] == b"\x89PNG"
    cropper.close()


def test_autocad_shx_comments_are_read_as_text():
    from l2c_rebar.pdf.text import page_text_lines

    doc = pymupdf.open()
    page = doc.new_page(width=400, height=200)
    page.draw_line((40, 60), (90, 60))  # stroke lettering stands in as plain line work
    note = page.add_text_annot((40, 50), "8-25M")
    note.set_info(title="AutoCAD SHX Text", content="8-25M")
    note.update()
    other = page.add_text_annot((200, 100), "review comment")
    other.set_info(title="Reviewer", content="12-15M")
    other.update()
    lines = page_text_lines(page, min_words=25)
    assert [l.text for l in lines] == ["8-25M"]
    assert abs(lines[0].bbox[0] - 40) < 2 and abs(lines[0].bbox[1] - 50) < 2


def test_a_page_the_ocr_cannot_read_does_not_stop_the_run(tmp_path, monkeypatch):
    import l2c_rebar.pdf.ocr as ocr

    class Broken:
        def read_page(self, page):
            raise RuntimeError("unreadable")

    monkeypatch.setattr(ocr, "_init_worker", lambda cfg, threads: setattr(ocr, "_WORKER_ENGINE", Broken()))
    pdf = tmp_path / "plot.pdf"
    doc = pymupdf.open()
    doc.new_page()
    doc.save(pdf)
    assert ocr.ocr_pages([(pdf, 0)], Config(workers=1), tmp_path / "cache") == 1
    assert not list((tmp_path / "cache").glob("*.json"))  # left out, so a later run tries it again
