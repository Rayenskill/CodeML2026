"""Investigate missing model structure using historical data only.

Tests smooth interactions, within-region merit normalization, legitimate
context, and correction strength. Aggregate previews motivate these hypotheses;
this script never reads preview results or reconstructed reference labels.
"""
import copy
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from model import ContextGrantModel, GrantModel, ROOT, digest, features, make_estimator, write_predictions
from train import metrics

HERE = Path(__file__).resolve().parent
OUT = HERE / "structural"


def main():
    OUT.mkdir(exist_ok=True)
    history = pd.read_csv(ROOT / "data/donnees_demandes.csv")
    x = features(history)
    y = history.decision_octroi.to_numpy()
    split = pd.read_csv(HERE / "artifacts/validation_split.csv")
    if not np.array_equal(split.id_candidat, history.id_candidat):
        raise ValueError("Historical data no longer match the audited split.")
    dev = np.flatnonzero(split.split.to_numpy() == "development")
    fold_id = split.cv_fold.to_numpy()[dev]
    folds = [(np.flatnonzero(fold_id != i), np.flatnonzero(fold_id == i)) for i in range(5)]
    configs = [dict(family="tensor", knots=k, C=c) for k in [3, 4, 5] for c in [0.01, 0.1, 1.0]]
    configs += [dict(family="quadratic", C=c) for c in [0.1, 1., 10.]]
    configs += [dict(family="regional_spline", standardize=s, C=c) for s in [False, True] for c in [0.3, 3.]]
    configs += [dict(family="context_spline", C=c) for c in [0.3, 3., 30.]]
    configs += [dict(family="sparse_spline", knots=5, curve_hours=True, C=3.)]
    records = []
    for config in configs:
        oof = np.zeros(len(dev))
        for train, valid in folds:
            fitted = make_estimator(config).fit(x.iloc[dev[train]], y[dev[train]])
            oof[valid] = fitted.predict_proba(x.iloc[dev[valid]])[:, 1]
        record = {"config": config, "development_cv": metrics(y[dev], oof)}
        records.append(record)
        print(json.dumps(record), flush=True)
        (OUT / "search_progress.json").write_text(json.dumps(records, indent=2) + "\n")
    records.sort(key=lambda r: r["development_cv"]["log_loss"])
    chosen = {f: next(r for r in records if r["config"]["family"] == f)
              for f in ["tensor", "quadratic", "regional_spline", "context_spline"]}
    chosen["regional_standardized"] = next(r for r in records if
        r["config"]["family"] == "regional_spline" and r["config"]["standardize"])
    models = {}
    for family in ["tensor", "quadratic"]:
        models[family] = GrantModel(chosen[family]["config"]).fit(history)
    models["regional_merit"] = ContextGrantModel(chosen["regional_spline"]["config"],
                                                retained_columns=("merit_group",)).fit(history)
    models["regional_standardized"] = ContextGrantModel(chosen["regional_standardized"]["config"],
                                                        retained_columns=("merit_group",)).fit(history)
    models["legitimate_context"] = ContextGrantModel(chosen["context_spline"]["config"],
        retained_columns=("first_generation", "programme")).fit(history)
    incumbent_config = dict(family="sparse_spline", knots=5, curve_hours=True, C=3.)
    incumbent = GrantModel(incumbent_config).fit(history)
    for retention in [0.05, 0.10, 0.15]:
        # Reuse the same historical fit: only the global mitigation strength
        # changes. Recomputing its training threshold remains training-only.
        model = ContextGrantModel(incumbent_config, remote_retention=retention)
        model.estimator_ = copy.deepcopy(incumbent.estimator_)
        model.profiles_ = incumbent.profiles_.copy(deep=True)
        model.training_rows_ = len(history)
        model.threshold_ = float(np.quantile(model.score(history), 0.6))
        models[f'region_correction_{round(100*(1-retention))}'] = model
    for name, model in models.items():
        joblib.dump(model, OUT / f"{name}.joblib", compress=3)
    report = {"training_rows": len(history), "development_rows": len(dev),
              "training_sha256": digest(ROOT / "data/donnees_demandes.csv"),
              "historical_holdout_used_for_selection": False,
              "preview_results_used_for_fitting": False, "reference_accuracy": None,
              "search": records, "selected": chosen,
              "retention_hypotheses": [0.05, 0.10, 0.15],
              "reference_claim": "No hidden rule has been established; these are testable feature-based hypotheses."}
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    applicants = pd.read_csv(ROOT / "data/candidats_evaluation.csv")
    if set(history.id_candidat) & set(applicants.id_candidat):
        raise ValueError("Training/evaluation ID overlap.")
    incumbent_pred = incumbent.allocate(applicants)
    rows = []
    for name, model in models.items():
        row = write_predictions(model, applicants, ROOT / "upload_model_round4" / f"{name}.csv")
        row.update(model=name, artifact=str((OUT / f"{name}.joblib").relative_to(ROOT)),
                   different_decisions_from_verified_sparse_model=int((model.allocate(applicants) != incumbent_pred).sum()),
                   reference_accuracy=None)
        rows.append(row)
    (OUT / "submissions.json").write_text(json.dumps(rows, indent=2) + "\n")
    print("SUBMISSIONS", json.dumps(rows), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
