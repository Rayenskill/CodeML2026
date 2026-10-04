"""Train reusable income-correction hypotheses using historical labels only.

Global retention and budget settings are explicit policy hypotheses. Historical
committee labels cannot validate the independent reference policy. No preview
results, earlier submissions or inferred reference labels are read here.
"""
import copy
import json
from pathlib import Path
import tempfile

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from model import ContextGrantModel, GrantModel, ROOT, digest, write_predictions

HERE = Path(__file__).resolve().parent


def main():
    out = HERE / "income_policy"
    out.mkdir(exist_ok=True)
    history = pd.read_csv(ROOT / "data/donnees_demandes.csv")
    config = dict(family="sparse_spline", knots=5, curve_hours=True, C=3.)
    base = GrantModel(config).fit(history)
    models = {}
    for name, retention in [("income_retained_05", .05), ("income_retained_15", .15),
                            ("income_need_05", -.05), ("income_need_15", -.15)]:
        model = ContextGrantModel(config, income_retention=retention)
        model.estimator_ = copy.deepcopy(base.estimator_)
        model.profiles_ = base.profiles_.copy(deep=True)
        model.training_rows_ = len(history)
        model.threshold_ = float(np.quantile(model.score(history), .6))
        models[name] = model
    for rate in [.39, .41]:
        model = copy.deepcopy(base)
        model.grant_rate = rate
        model.threshold_ = float(np.quantile(model.score(history), 1-rate))
        models[f"budget_{round(rate*100)}"] = model
    for name, model in models.items():
        joblib.dump(model, out / f"{name}.joblib", compress=3)
    applicants = pd.read_csv(ROOT / "data/candidats_evaluation.csv")
    if set(history.id_candidat) & set(applicants.id_candidat):
        raise ValueError("Training and evaluation IDs overlap.")
    baseline = base.allocate(applicants)
    records = []
    for name in models:
        artifact = out / f"{name}.joblib"
        model = joblib.load(artifact)
        path = ROOT / "upload_model_round5" / f"{name}.csv"
        record = write_predictions(model, applicants, path)
        with tempfile.TemporaryDirectory() as temp:
            regenerated = Path(temp) / "predictions.csv"
            write_predictions(joblib.load(artifact), applicants, regenerated)
            assert regenerated.read_bytes() == path.read_bytes()
        sample = applicants.iloc[:12].copy()
        scores = model.score(sample)
        sample.id_candidat = [f"NEW{i}" for i in range(len(sample))]
        sample["decision_octroi"] = 1
        np.testing.assert_allclose(scores, model.score(sample), rtol=0, atol=1e-13)
        np.testing.assert_allclose(scores, model.score(sample.iloc[::-1])[::-1], rtol=0, atol=1e-13)
        np.testing.assert_allclose(scores[:1], model.score(sample.iloc[:1]), rtol=0, atol=1e-13)
        record.update(model=name, artifact=str(artifact.relative_to(ROOT)),
                      artifact_sha256=digest(artifact),
                      income_retention=getattr(model, "income_retention", 0.),
                      different_decisions_from_incumbent=int((model.allocate(applicants) != baseline).sum()),
                      reference_accuracy=None, verification_passed=True)
        records.append(record)
        print(json.dumps(record), flush=True)
    (out / "submissions.json").write_text(json.dumps(records, indent=2)+"\n")
    (out / "provenance.json").write_text(json.dumps(dict(
        historical_training_sha256=digest(ROOT / "data/donnees_demandes.csv"),
        training_rows=len(history), config=config, reference_labels_used=False,
        preview_results_read=False, policy_parameters_selected_on_historical_target=False,
        limitation="Policy hypotheses require platform scoring; historical labels are a different target."
    ), indent=2)+"\n")


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
