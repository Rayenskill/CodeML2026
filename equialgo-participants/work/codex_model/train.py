"""Choose a committee estimator by training-only CV, then fit a corrected model.

No imports or reads from earlier submissions, score reports, or reconstructed
labels. The sole label source is data/donnees_demandes.csv.
"""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from threadpoolctl import threadpool_limits

from model import GrantEnsemble, GrantModel, ROOT, SEED, digest, features, make_estimator, write_predictions

HERE = Path(__file__).resolve().parent


def candidates() -> list[dict]:
    specs = []
    for reduced in [True, False]:
        for c in [0.1, 1.0, 100.0]:
            specs.append(dict(family="linear", reduced=reduced, C=c))
    for all_splines in [False, True]:
        for knots in [3, 5, 7]:
            for c in [0.03, 0.3, 3.0]:
                specs.append(dict(family="spline", all_splines=all_splines, knots=knots, C=c))
    for leaves, min_leaf, l2, iterations in [
        (7, 80, 10, 200), (7, 80, 30, 400), (15, 80, 10, 200),
        (15, 80, 30, 400), (15, 30, 10, 200), (31, 40, 30, 200),
    ]:
        specs.append(dict(family="boosted", leaves=leaves, min_leaf=min_leaf,
                          l2=l2, iterations=iterations))
    specs.append(dict(family="forest"))
    return specs


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    label = np.asarray(p) >= 0.5
    n, k = len(y), int((y == label).sum())
    # Wilson 95% interval for agreement with historical committee labels only.
    z = 1.959963984540054
    centre = (k / n + z * z / (2 * n)) / (1 + z * z / n)
    radius = z * np.sqrt((k / n) * (1 - k / n) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return {"accuracy": float(accuracy_score(y, label)),
            "accuracy_wilson_95_interval": [float(centre - radius), float(centre + radius)],
            "macro_f1": float(f1_score(y, label, average="macro")),
            "roc_auc": float(roc_auc_score(y, p)), "log_loss": float(log_loss(y, p)),
            "brier": float(brier_score_loss(y, p))}


def fit_weights(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    start = np.ones(p.shape[1]) / p.shape[1]
    solution = minimize(lambda w: log_loss(y, p @ w), start, method="SLSQP",
                        bounds=[(0, 1)] * len(start),
                        constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
                        options={"maxiter": 300, "ftol": 1e-10})
    if not solution.success:
        raise RuntimeError(f"Ensemble weight fitting failed: {solution.message}")
    weights = np.clip(solution.x, 0, 1)
    return weights / weights.sum()


def build_model(config: dict) -> GrantModel:
    if config["family"] == "ensemble":
        return GrantEnsemble(config["members"], config["weights"])
    return GrantModel(config)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", type=Path, default=ROOT / "data/donnees_demandes.csv")
    parser.add_argument("--artifacts", type=Path, default=HERE / "artifacts")
    parser.add_argument("--candidates", type=Path, default=None,
                        help="Optional applicant file to score, read only AFTER training is complete.")
    parser.add_argument("--output", type=Path, default=ROOT / "upload_model")
    args = parser.parse_args()
    args.artifacts.mkdir(parents=True, exist_ok=True)
    history = pd.read_csv(args.training)
    if not history.id_candidat.is_unique or history.id_candidat.isna().any():
        raise ValueError("Historical IDs must be unique and nonmissing.")
    if not history.decision_octroi.isin([0, 1]).all():
        raise ValueError("Historical labels must be binary.")
    # Group identical feature vectors before splitting if future training data
    # include duplicates. Reject here rather than silently leaking duplicates.
    x = features(history)
    if x.duplicated().any():
        raise ValueError("Duplicate feature rows require grouped validation; cannot use this split.")
    y = history.decision_octroi.to_numpy(dtype=int)
    dev, test = train_test_split(np.arange(len(history)), test_size=0.20,
                                stratify=y, random_state=SEED)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED + 1).split(dev, y[dev]))
    split = pd.DataFrame({"id_candidat": history.id_candidat, "split": "development", "cv_fold": -1})
    split.loc[test, "split"] = "holdout"
    for fold, (_, valid) in enumerate(folds):
        split.loc[dev[valid], "cv_fold"] = fold
    split.to_csv(args.artifacts / "validation_split.csv", index=False)
    records = []
    oof = {}
    specs = candidates()
    print(f"{len(history)} historical rows; {len(dev)} development, {len(test)} holdout; "
          f"{len(specs)} configurations, 5 folds", flush=True)
    with threadpool_limits(limits=2):
        for number, config in enumerate(specs):
            out_of_fold = np.zeros(len(dev))
            fold_losses = []
            for train, valid in folds:
                estimator = make_estimator(config).fit(x.iloc[dev[train]], y[dev[train]])
                p = estimator.predict_proba(x.iloc[dev[valid]])[:, 1]
                out_of_fold[valid] = p
                fold_losses.append(float(log_loss(y[dev[valid]], p)))
            row = {"number": number, "config": config, "cv": metrics(y[dev], out_of_fold),
                   "fold_log_loss": fold_losses}
            records.append(row)
            oof[number] = out_of_fold
            print(json.dumps(row), flush=True)
            (args.artifacts / "search_progress.json").write_text(json.dumps(records, indent=2) + "\n")
        # Tune mixture weights with out-of-fold predictions, then evaluate the
        # meta-model out of fold too. The reserved holdout stays untouched.
        members = [min((r for r in records if r["config"]["family"] == family),
                       key=lambda r: r["cv"]["log_loss"])
                   for family in ["linear", "spline", "boosted"]]
        matrix = np.column_stack([oof[r["number"]] for r in members])
        for weighted in [False, True]:
            meta_oof = np.zeros(len(dev))
            for train, valid in folds:
                weights = fit_weights(matrix[train], y[dev[train]]) if weighted else np.ones(3) / 3
                meta_oof[valid] = matrix[valid] @ weights
            weights = fit_weights(matrix, y[dev]) if weighted else np.ones(3) / 3
            config = {"family": "ensemble", "weighted": weighted,
                      "members": [r["config"] for r in members], "weights": weights.tolist()}
            row = {"number": len(records), "config": config, "cv": metrics(y[dev], meta_oof),
                   "fold_log_loss": [float(log_loss(y[dev[v]], meta_oof[v])) for _, v in folds]}
            records.append(row)
            print("ENSEMBLE", json.dumps(row), flush=True)
        # Both model choices are frozen before accessing holdout outcomes.
        records.sort(key=lambda r: (r["cv"]["log_loss"], r["number"]))
        winner = records[0]
        best_single = next(r for r in records if r["config"]["family"] != "ensemble")
        best_ensemble = next(r for r in records if r["config"]["family"] == "ensemble")
        development_model = build_model(winner["config"]).fit(history.iloc[dev])
        holdout_p = development_model.committee_probability(history.iloc[test])
        holdout_metrics = metrics(y[test], holdout_p)
        corrected_holdout = development_model.allocate(history.iloc[test])
        corrected_diagnostics = {"agreement_with_historical_committee": float(accuracy_score(y[test], corrected_holdout)),
                                 "grant_rate": float(corrected_holdout.mean()),
                                 "not_hidden_reference_accuracy": True}
        known = pd.DataFrame({"id_candidat": history.iloc[test].id_candidat,
                              "historical_label": y[test], "committee_probability": holdout_p,
                              "corrected_allocation": corrected_holdout})
        known.to_csv(args.artifacts / "holdout_predictions.csv", index=False)
        holdout_comparison = []
        for label, choice in [("best_single", best_single), ("best_ensemble", best_ensemble)]:
            fitted = development_model if choice["number"] == winner["number"] else build_model(choice["config"]).fit(history.iloc[dev])
            holdout_comparison.append({"model": label, "config": choice["config"],
                "historical_committee_metrics": metrics(y[test], fitted.committee_probability(history.iloc[test]))})
        # Final fitting uses all historical rows after validation and selection.
        final_model = build_model(winner["config"]).fit(history)
        joblib.dump(final_model, args.artifacts / "model.joblib", compress=3)
        alternatives = []
        for family in ["linear", "spline", "boosted", "ensemble"]:
            best = next(r for r in records if r["config"]["family"] == family)
            fitted = final_model if best["number"] == winner["number"] else build_model(best["config"]).fit(history)
            name = f"{family}.joblib"
            joblib.dump(fitted, args.artifacts / name, compress=3)
            alternatives.append({"family": family, "config": best["config"], "file": name, "cv": best["cv"]})
        report = {
            "training_file": str(args.training.resolve()), "training_sha256": digest(args.training),
            "training_rows": len(history), "development_rows": len(dev), "holdout_rows": len(test),
            "seed": SEED, "selection_metric": "5-fold development log loss against historical committee labels",
            "selected": winner, "historical_committee_holdout": holdout_metrics,
            "holdout_comparison": holdout_comparison,
            "corrected_holdout_diagnostics": corrected_diagnostics,
            "reference_accuracy": None, "reference_accuracy_status": "Unmeasured; requires external scoring",
            "score_inputs": ["cote_r_equivalent", "heures_travail_semaine"],
            "mitigation_assumption": "Retain learned academic/work effects; neutralize other committee effects with common training profiles.",
            "evaluation_data_used_for_training_or_selection": False,
            "preview_readings_or_reconstructed_labels_used": False,
            "model_sha256": digest(args.artifacts / "model.joblib"),
            "alternatives": alternatives, "search": records,
            "versions": {"python": platform.python_version(), "numpy": np.__version__,
                         "scipy": scipy.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__},
        }
        (args.artifacts / "training_report.json").write_text(json.dumps(report, indent=2) + "\n")
        print("SELECTED", json.dumps(winner), flush=True)
        print("HISTORICAL_HOLDOUT", json.dumps(holdout_metrics), flush=True)
        if args.candidates is not None:
            applicants = pd.read_csv(args.candidates)
            if set(history.id_candidat) & set(applicants.id_candidat):
                raise ValueError("Refusing to score training applicants as an independent evaluation cohort.")
            submissions = [write_predictions(final_model, applicants, args.output / "predictions.csv")]
            for alt in alternatives:
                fitted = joblib.load(args.artifacts / alt["file"])
                submissions.append(write_predictions(fitted, applicants, args.output / f'{alt["family"]}.csv'))
            (args.artifacts / "submissions.json").write_text(json.dumps(submissions, indent=2) + "\n")
            print("SUBMISSIONS", json.dumps(submissions), flush=True)


if __name__ == "__main__":
    main()
