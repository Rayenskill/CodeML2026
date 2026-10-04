"""Tests du livrable JADCO : le notebook exécuté, les contrôles de validité, la qualité et la confidentialité.

Lancer :  python -m pytest tests -q        (depuis work/final, après avoir exécuté le notebook)
"""
import json
import re
from pathlib import Path

import nbformat
import pytest

FINAL_DIR = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = FINAL_DIR / "equinoxe_hausse_2026.ipynb"
CRM_FILE_NAMES = {"equinoxe_listings.csv", "equinoxe_lease_history.csv", "equinoxe_concessions.csv", "equinoxe_asking_history.csv"}
EXPECTED_VALIDITY_SECTIONS = ("section 2", "section 4", "section 6", "section 7", "section 8", "section 9", "section 10", "section 10 (suite)", "section 11")
MIN_BACKTEST_YEARS = 3
MAX_PLAUSIBLE_GROWTH_PCT = 25


@pytest.fixture(scope="module")
def notebook():
    return nbformat.read(NOTEBOOK_PATH, as_version=4)


@pytest.fixture(scope="module")
def all_output_text(notebook):
    chunks = []
    for cell in notebook.cells:
        for output in cell.get("outputs", []) if cell.cell_type == "code" else []:
            chunks.append(output.get("text") or output.get("data", {}).get("text/plain") or "")
    return "\n".join(chunks)


def test_every_code_cell_executed_without_error(notebook):
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != "code" or not cell.source.strip():
            continue
        assert cell.execution_count is not None, f"cellule {index} non exécutée"
        assert not any(output.get("output_type") == "error" for output in cell.get("outputs", [])), f"erreur à la cellule {index}"


@pytest.mark.parametrize("section", EXPECTED_VALIDITY_SECTIONS)
def test_validity_checks_printed_ok(all_output_text, section):
    assert f"Contrôles de la {section} : OK" in all_output_text


def test_lint_gate_passes():
    from lint_notebook import lint
    findings, _ = lint(str(NOTEBOOK_PATH))
    assert findings == [], "\n".join(str(finding) for finding in findings)


def test_final_answer_is_consistent():
    answer = json.loads((FINAL_DIR / "outputs" / "final_answer.json").read_text(encoding="utf-8"))
    low, high = answer["headline_band_pct"]
    assert low <= answer["headline_effective_pct"] <= high
    assert answer["contract_band_pct"][0] <= answer["contract_pct"] <= answer["contract_band_pct"][1]
    assert abs(answer["headline_effective_pct"]) < MAX_PLAUSIBLE_GROWTH_PCT and abs(answer["contract_pct"]) < MAX_PLAUSIBLE_GROWTH_PCT
    scenarios = answer["scenarios_effective_pct"]
    assert scenarios["bas (cliquet)"] < scenarios["coin indexé + pourcentage de base"] < scenarios["haut (ancienne méthode)"]
    assert "effectif" in answer["definition"] and "unité constante" in answer["definition"]


def test_backtest_covers_required_years(all_output_text):
    for year in (2023, 2024, 2025):
        assert re.search(rf"^{year}\s", all_output_text, re.MULTILINE), f"année {year} absente du backtest"


def test_no_crm_rows_in_outputs():
    for csv_file in (FINAL_DIR / "outputs").glob("*.csv"):
        header = csv_file.read_text(encoding="utf-8").splitlines()[0].split(",")
        for forbidden in ("unit_code", "unit_id", "lease_id", "sign_date", "lease_start", "rent_contract", "rent_effective"):
            assert forbidden not in header, f"{csv_file.name} contient {forbidden}"


def test_no_crm_source_file_in_deliverable():
    found = {path.name for path in FINAL_DIR.rglob("*.csv")} & CRM_FILE_NAMES
    assert not found, f"fichiers CRM présents dans le livrable : {found}"


def test_notebook_is_not_a_data_dump():
    size_megabytes = NOTEBOOK_PATH.stat().st_size / 1e6
    assert size_megabytes < 5, "le notebook ne doit pas embarquer de données (taille anormale)"


def test_required_functions_and_sections_exist(notebook):
    source = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
    assert "def estimate_2026(leases, asking, external=None)" in source
    assert "def backtest(leases, target_year" in source
    markdown = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "markdown")
    for heading in ("## 13. Références", "## 12. Limites", "Outils d'intelligence artificielle"):
        assert heading in markdown


def test_public_data_has_provenance():
    import pandas as pd
    external = pd.read_csv(FINAL_DIR / "external" / "tidy" / "external_annual.csv")
    for column in ("source_url", "retrieved_on", "published_on", "confidence"):
        assert external[column].notna().all(), f"{column} manquant"


def test_shipped_notebook_matches_current_sources(notebook):
    from build_notebook import build
    parts = sorted(int(path.stem.removeprefix("nb_part")) for path in FINAL_DIR.glob("nb_part*.py"))
    expected = build(parts)
    assert [(cell.cell_type, cell.source) for cell in notebook.cells] == [
        (cell.cell_type, cell.source) for cell in expected.cells
    ], "Notebook périmé : reconstruire et exécuter les sources avant livraison"


def test_notice_windows_use_calendar_months():
    import ast
    from types import SimpleNamespace

    import numpy as np
    import pandas as pd

    tree = ast.parse((FINAL_DIR / "nb_part09.py").read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "notice_share_before")
    settings = SimpleNamespace(notice_months_long_lease=(3, 6), notice_months_short_lease=(1, 2), notice_timing="uniform")
    namespace = {"pd": pd, "np": np, "CONFIG": SimpleNamespace(months_per_year=12), "FORECAST": settings}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "notice_window", "exec"), namespace)
    cohort = pd.DataFrame({"term_months": [12, 12, 12, 6],
                           "lease_end": pd.to_datetime(["2026-03-31", "2026-06-30", "2026-09-30", "2026-02-28"])})
    share = namespace["notice_share_before"](cohort, pd.Timestamp("2026-01-01"))
    # June 30: legal window December 30 to March 30; two days fall before reform.
    # Short lease ending February 28: December 28 to January 28, four days before reform.
    np.testing.assert_allclose(share, [1, 2 / 90, 0, 4 / 31])
    settings.notice_timing = "earliest"
    np.testing.assert_array_equal(namespace["notice_share_before"](cohort, pd.Timestamp("2026-01-01")), [1, 1, 0, 1])
    settings.notice_timing = "latest"
    np.testing.assert_array_equal(namespace["notice_share_before"](cohort, pd.Timestamp("2026-01-01")), [1, 0, 0, 0])
