"""The Streamlit interface, driven headless on the synthetic project."""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from l2c_rebar.synthetic import build_project  # noqa: E402
from l2c_rebar.validation import REJECTED, apply_validations, load_validations, result_key, save_validations  # noqa: E402

APP = str(Path(__file__).parents[1] / "src" / "l2c_rebar" / "app.py")


def test_app_starts_and_waits_for_a_project():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert "Indiquez le dossier" in at.info[0].value


def test_app_runs_a_project_and_lists_findings(tmp_path):
    project = build_project(tmp_path)
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.text_input[0].set_value(str(project))
    at.text_input[1].set_value(str(tmp_path / "out"))
    at.selectbox[0].set_value("off")
    at.checkbox[0].set_value(False).run()
    at.button[0].click().run()
    assert not at.exception
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Non conforme"] == "5"
    assert (tmp_path / "out" / "DEMO" / "DEMO_elements.json").exists()


def test_rejected_findings_leave_the_report(tmp_path):
    from l2c_rebar.config import Config
    from l2c_rebar.pipeline import run_project

    out = run_project(build_project(tmp_path), tmp_path / "out", Config(ocr="off"))
    target = next(r for r in out.results if r.statut == "non_conforme")
    path = tmp_path / "validations.json"
    save_validations(path, {result_key(target): {"verdict": REJECTED, "commentaire": "fausse alerte"}})
    kept = apply_validations(list(out.results), load_validations(path))
    assert target not in kept and len(kept) == len(out.results) - 1
