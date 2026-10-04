"""Refine a reusable spline classifier; no applicant-specific corrections.

All fitted coefficients and blend weights use historical labels and development
folds only. Preview feedback chooses this model family as the next experiment,
but no preview scores, old predictions, or reconstructed labels are read here.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from model import GrantEnsemble, GrantModel, ROOT, digest, features, make_estimator, write_predictions
from train import fit_weights, metrics

HERE = Path(__file__).resolve().parent
OUT = HERE / "spline_tuning"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    history = pd.read_csv(ROOT / "data/donnees_demandes.csv")
    x = features(history)
    y = history.decision_octroi.to_numpy()
    split = pd.read_csv(HERE / "artifacts/validation_split.csv")
    if not np.array_equal(split.id_candidat, history.id_candidat):
        raise ValueError("The audited historical split no longer matches.")
    dev = np.flatnonzero(split.split.to_numpy() == "development")
    fold_id = split.cv_fold.to_numpy()[dev]
    folds = [(np.flatnonzero(fold_id != i), np.flatnonzero(fold_id == i)) for i in range(5)]
    configs = [dict(family="sparse_spline", knots=k, curve_hours=True, C=c)
               for k in [3, 4, 5, 6, 7, 9] for c in [1.0, 3.0, 10.0, 30.0]]
    configs += [dict(family="sparse_spline", knots=k, curve_hours=True, C=c,
                    knot_spacing="uniform") for k in [4, 5, 7] for c in [1.0, 10.0]]
    results, probabilities = [], {}
    for number, config in enumerate(configs):
        oof = np.zeros(len(dev))
        for train, valid in folds:
            fitted = make_estimator(config).fit(x.iloc[dev[train]], y[dev[train]])
            oof[valid] = fitted.predict_proba(x.iloc[dev[valid]])[:, 1]
        record = {"number": number, "config": config, "development_cv": metrics(y[dev], oof)}
        results.append(record)
        probabilities[number] = oof
        print(json.dumps(record), flush=True)
        (OUT / "search_progress.json").write_text(json.dumps(results, indent=2) + "\n")
    results.sort(key=lambda r: r["development_cv"]["log_loss"])
    # Distinct smoothness choices are retained as explicit model hypotheses.
    # Each group's regularization is selected by historical development loss.
    choices = {
        "smooth": next(r for r in results if r["config"]["knots"] <= 4 and r["config"].get("knot_spacing", "quantile") == "quantile"),
        "flexible": next(r for r in results if r["config"]["knots"] >= 6 and r["config"].get("knot_spacing", "quantile") == "quantile"),
        "uniform": next(r for r in results if r["config"].get("knot_spacing") == "uniform"),
        "regularized": next(r for r in results if r["config"]["knots"] == 5 and r["config"]["C"] < 3 and r["config"].get("knot_spacing", "quantile") == "quantile"),
        "less_regularized": next(r for r in results if r["config"]["knots"] == 5 and r["config"]["C"] > 3 and r["config"].get("knot_spacing", "quantile") == "quantile"),
    }
    members = [choices[name] for name in ["smooth", "flexible", "uniform"]]
    matrix = np.column_stack([probabilities[r["number"]] for r in members])
    blend_oof = np.zeros(len(dev))
    for train, valid in folds:
        weights = fit_weights(matrix[train], y[dev[train]])
        blend_oof[valid] = matrix[valid] @ weights
    blend_weights = fit_weights(matrix, y[dev])
    blend_record = {"configs": [r["config"] for r in members], "weights": blend_weights.tolist(),
                    "development_cv": metrics(y[dev], blend_oof)}
    print("BLEND", json.dumps(blend_record), flush=True)
    # Fit only to historical features/labels; evaluation has not been loaded.
    fitted = {name: GrantModel(record["config"]).fit(history) for name, record in choices.items()}
    fitted["spline_blend"] = GrantEnsemble(blend_record["configs"], blend_record["weights"]).fit(history)
    # Refit the known model to make the threshold experiment reproducible from
    # historical data alone, without loading any earlier predictions/artifact.
    incumbent = GrantModel(dict(family="sparse_spline", knots=5, curve_hours=True, C=3.0)).fit(history)
    fitted["training_threshold"] = copy.deepcopy(incumbent)
    fitted["training_threshold"].allocation_rule = "training_threshold"
    for name, model in fitted.items():
        joblib.dump(model, OUT / f"{name}.joblib", compress=3)
    report = {
        "training_file": "data/donnees_demandes.csv", "training_sha256": digest(ROOT / "data/donnees_demandes.csv"),
        "training_rows": len(history), "development_rows": len(dev),
        "historical_holdout_used_for_selection": False, "preview_scores_used_in_fitting": False,
        "reference_accuracy": None, "search": results, "selected": choices, "blend": blend_record,
        "family_choice_context": "User reported sparse spline 94.88%; this determines the family to investigate, not fitted coefficients.",
        "training_threshold": {"training_quantile": 0.6, "threshold": incumbent.threshold_},
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    applicants = pd.read_csv(ROOT / "data/candidats_evaluation.csv")
    if set(applicants.id_candidat) & set(history.id_candidat):
        raise ValueError("Historical/evaluation ID overlap.")
    output = ROOT / "upload_model_round3"
    output.mkdir(exist_ok=True)
    submissions, seen = [], {}
    incumbent_predictions = incumbent.allocate(applicants)
    for name, model in fitted.items():
        pred = model.allocate(applicants)
        different = int(np.sum(pred != incumbent_predictions))
        duplicate = seen.get(pred.tobytes())
        info = {"model": name, "artifact": str((OUT / f"{name}.joblib").relative_to(ROOT)),
                "different_decisions_from_previous_sparse_model": different,
                "duplicate_of": duplicate, "reference_accuracy": None}
        if different == 0:
            info["skipped"] = "Same decisions as the already scored sparse spline."
        elif duplicate is not None:
            info["skipped"] = "Same decisions as another new model."
        else:
            info.update(write_predictions(model, applicants, output / f"{name}.csv"))
            seen[pred.tobytes()] = name
        submissions.append(info)
    (OUT / "submissions.json").write_text(json.dumps(submissions, indent=2) + "\n")
    print("SUBMISSIONS", json.dumps(submissions), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
