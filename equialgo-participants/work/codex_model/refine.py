"""Training-only refinement: sparse smooth effects, monotonic trees, and bagging.

Uses the same development folds. It does not revisit holdout outcomes or use
preview observations to fit coefficients, choose weights, or edit applicants.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from threadpoolctl import threadpool_limits

from model import BaggedGrantModel, GrantModel, ROOT, SEED, digest, features, make_estimator, write_predictions
from train import metrics

HERE = Path(__file__).resolve().parent
OUT = HERE / "refinement"


def main():
    OUT.mkdir(exist_ok=True)
    history = pd.read_csv(ROOT / "data/donnees_demandes.csv")
    x = features(history)
    y = history.decision_octroi.to_numpy()
    split = pd.read_csv(HERE / "artifacts/validation_split.csv")
    if not np.array_equal(split.id_candidat, history.id_candidat):
        raise ValueError("Training data no longer match the audited split.")
    dev = np.flatnonzero(split.split.to_numpy() == "development")
    fold_id = split.cv_fold.to_numpy()[dev]
    folds = [(np.flatnonzero(fold_id != i), np.flatnonzero(fold_id == i)) for i in range(5)]
    configs = [dict(family="sparse_spline", knots=5, curve_hours=curve_hours, C=c)
               for curve_hours in [False, True] for c in [0.03, 0.3, 3.0]]
    configs += [dict(family="monotonic", leaves=leaves, iterations=iterations, l2=l2)
                for leaves, iterations, l2 in [(7, 150, 10), (7, 300, 30), (15, 150, 10), (15, 300, 30)]]
    results = []
    for config in configs:
        oof = np.zeros(len(dev))
        for train, valid in folds:
            fitted = make_estimator(config).fit(x.iloc[dev[train]], y[dev[train]])
            oof[valid] = fitted.predict_proba(x.iloc[dev[valid]])[:, 1]
        record = {"config": config, "development_cv": metrics(y[dev], oof)}
        results.append(record)
        print(json.dumps(record), flush=True)
        (OUT / "search_progress.json").write_text(json.dumps(results, indent=2) + "\n")
    spline = min((r for r in results if r["config"]["family"] == "sparse_spline"),
                 key=lambda r: r["development_cv"]["log_loss"])
    monotonic = min((r for r in results if r["config"]["family"] == "monotonic"),
                    key=lambda r: r["development_cv"]["log_loss"])
    config = spline["config"]
    # Evaluate an equal-weight bagged predictor on the same development folds.
    oof = np.zeros(len(dev))
    for fold, (train, valid) in enumerate(folds):
        rng = np.random.default_rng(SEED + 8 + fold)
        for _ in range(20):
            sample = dev[train[rng.integers(0, len(train), size=len(train))]]
            fitted = make_estimator(config).fit(x.iloc[sample], y[sample])
            oof[valid] += fitted.predict_proba(x.iloc[dev[valid]])[:, 1] / 20
    bagged_record = {"config": {"family": "bagging", "base": config, "n_estimators": 20},
                     "development_cv": metrics(y[dev], oof)}
    results.append(bagged_record)
    print("BAGGED", json.dumps(bagged_record), flush=True)
    models = {
        "sparse_spline": GrantModel(spline["config"]).fit(history),
        "monotonic": GrantModel(monotonic["config"]).fit(history),
        "bagged_spline": BaggedGrantModel(config, 20).fit(history),
    }
    for name, model in models.items():
        joblib.dump(model, OUT / f"{name}.joblib", compress=3)
    report = {"training_file": "data/donnees_demandes.csv", "training_sha256": digest(ROOT / "data/donnees_demandes.csv"),
              "training_rows": len(history), "development_rows": len(dev),
              "holdout_reused_for_selection": False,
              "preview_scores_used_in_fitting_or_selection": False,
              "reference_accuracy": None, "search": results,
              "selected_sparse": spline, "selected_monotonic": monotonic, "bagged": bagged_record}
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    # Read the evaluation cohort only after fitting and saving all models.
    applicants = pd.read_csv(ROOT / "data/candidats_evaluation.csv")
    if set(applicants.id_candidat) & set(history.id_candidat):
        raise ValueError("Training/evaluation ID overlap.")
    submissions = []
    for name, model in models.items():
        submissions.append(write_predictions(model, applicants, ROOT / "upload_model_round2" / f"{name}.csv"))
    (OUT / "submissions.json").write_text(json.dumps(submissions, indent=2) + "\n")
    print("SUBMISSIONS", json.dumps(submissions), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
