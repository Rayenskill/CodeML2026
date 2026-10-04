"""Numerical and information-boundary tests, independent of stored notebook outputs."""
import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def model_functions():
    tree = ast.parse((ROOT / "nb_part09.py").read_text())
    names = {"weighted_percentile", "concession_paths", "new_lease_drags", "forecast_mixture", "simulate_cohort", "known_at"}
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {
        "np": np, "pd": pd,
        "CONFIG": SimpleNamespace(random_seed=2026, days_per_year=365.25),
        "MIXTURE": SimpleNamespace(concession_paths=("indexed", "half", "ratchet"),
                                   renewal_readings=("base", "timed", "old")),
        "year_end": lambda year: pd.Timestamp(f"{year}-12-31"),
    }
    exec(compile(ast.Module(body=functions, type_ignores=[]), "model_contracts", "exec"), namespace)
    return namespace


def test_weighted_percentile_ignores_zero_mass_and_respects_endpoints(model_functions):
    percentile = model_functions["weighted_percentile"]
    assert percentile([99, 10, -99, 20], [0, 1, 0, 3], 0) == 10
    assert percentile([99, 10, -99, 20], [0, 1, 0, 3], 50) == 20
    assert percentile([99, 10, -99, 20], [0, 1, 0, 3], 100) == 20
    assert percentile([20, 10], [300, 100], 50) == 20


@pytest.mark.parametrize("values,weights,percentile", [
    ([], [], 50), ([1], [0], 50), ([1, 2], [1], 50),
    ([np.nan], [1], 50), ([1], [-1], 50), ([1], [1], 101),
])
def test_weighted_percentile_rejects_invalid_inputs(model_functions, values, weights, percentile):
    with pytest.raises(ValueError):
        model_functions["weighted_percentile"](values, weights, percentile)


def test_concession_paths_respect_ratchet_and_true_midpoint(model_functions):
    paths = model_functions["concession_paths"]
    assert paths(7, 9) == {"indexed": 7, "half": 8, "ratchet": 9}
    assert paths(10, 9) == {"indexed": 10, "half": 10, "ratchet": 10}


def test_zero_concessions_do_not_require_a_positive_depth_pool(model_functions):
    actual = model_functions["new_lease_drags"]({}, pd.DataFrame(index=range(2)), 2026, 0, 3, np.random.default_rng(0))
    np.testing.assert_array_equal(actual, np.zeros((3, 2)))


@pytest.mark.parametrize("priors", [({"bad": 1}, {"indexed": 1}), ({"timed": -1}, {"indexed": 1}),
                                         ({"timed": .5}, {"indexed": 1}), ({"timed": 1}, {})])
def test_forecast_rejects_invalid_policy_weights_before_computing(model_functions, priors):
    with pytest.raises(ValueError):
        model_functions["forecast_mixture"](None, None, None, 2026, "october", priors=priors)


def test_future_leases_and_asking_rows_cannot_enter_historical_context(model_functions):
    dates = pd.to_datetime(["2024-12-31", "2025-01-01"])
    leases = pd.DataFrame({"lease_start": dates, "marker": ["known", "future"]})
    asking = pd.DataFrame({"ask_date": dates, "marker": ["known", "future"]})
    known_leases, known_asking = model_functions["known_at"](leases, asking, 2025)
    assert known_leases.marker.tolist() == known_asking.marker.tolist() == ["known"]
    assert len(leases) == len(asking) == 2  # Filtering leaves caller data intact.


def test_simulated_growth_matches_exact_annualisation_for_long_and_short_leases(model_functions):
    cohort = pd.DataFrame({"lease_start": pd.to_datetime(["2024-01-01", "2025-07-01"]),
                           "next_start": pd.to_datetime(["2026-01-01", "2026-01-01"]),
                           "drag": [.2, .2], "province": ["Quebec", "Ontario"], "rent_contract": [1000, 1000]})
    context = {"pairs": pd.DataFrame({"lease_year": [2025, 2025], "is_renewal": [0, 1], "growth_contract_pct": [0, 0]})}
    model_functions["renewal_probability"] = lambda *args: np.ones(2)
    model_functions["new_lease_drags"] = lambda *args: np.full((3, 2), .1)
    simulation = model_functions["simulate_cohort"](context, cohort, 2026, 10, 10, 10, 3, 42)
    elapsed_years = np.array([731, 184]) / 365.25
    # Annual rent growth compounds over elapsed time; concession changes affect the rent ratio once.
    effective_rent_ratio = np.power(1.1, elapsed_years) * .9 / .8
    expected = (np.power(effective_rent_ratio, 1 / elapsed_years) - 1) * 100
    np.testing.assert_allclose(simulation["effective"], np.tile(expected, (3, 1)))
    np.testing.assert_allclose(simulation["contract"], 10)


def test_failed_build_preserves_the_last_good_notebook_and_outputs(tmp_path, monkeypatch):
    import build_notebook
    import nbformat

    (tmp_path / "outputs").mkdir()
    answer = tmp_path / "outputs/final_answer.json"
    answer.write_text('{"previous": true}')
    notebook_path = tmp_path / "delivered.ipynb"
    notebook_path.write_text("previous notebook")

    def fail_execute(client, **kwargs):
        staged_outputs = Path(kwargs["env"]["JADCO_OUTPUT_DIR"])
        staged_outputs.mkdir()
        (staged_outputs / "final_answer.json").write_text('{"partial": true}')
        raise RuntimeError("deliberate kernel failure")

    monkeypatch.setattr(build_notebook, "HERE", tmp_path)
    monkeypatch.setattr(build_notebook.NotebookClient, "execute", fail_execute)
    with pytest.raises(RuntimeError, match="deliberate kernel failure"):
        build_notebook.execute_and_publish(nbformat.v4.new_notebook(), notebook_path)
    assert answer.read_text() == '{"previous": true}'
    assert notebook_path.read_text() == "previous notebook"
    assert notebook_path.with_suffix(".failed.ipynb").exists()
