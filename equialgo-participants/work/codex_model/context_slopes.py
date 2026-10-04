"""Evaluate context-dependent committee merit slopes with training-only CV."""
import json
from pathlib import Path
import tempfile

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from model import GrantEnsemble, GrantModel, ROOT, digest, features, make_estimator, write_predictions
from train import metrics

HERE = Path(__file__).resolve().parent


def main():
    out = HERE / "context_slopes"
    out.mkdir(exist_ok=True)
    history = pd.read_csv(ROOT / "data/donnees_demandes.csv")
    x, y = features(history), history.decision_octroi.to_numpy()
    split = pd.read_csv(HERE / "artifacts/validation_split.csv")
    assert np.array_equal(split.id_candidat, history.id_candidat)
    dev = np.flatnonzero(split.split.to_numpy() == "development")
    fold_ids = split.cv_fold.to_numpy()[dev]
    baseline = dict(family="sparse_spline", knots=5, curve_hours=True, C=3.)
    configs = [dict(family="context_slopes", context=context, C=c)
               for context in [["remote"],["log_income"],["remote","log_income"]]
               for c in [.03,.3,3.]] + [baseline]
    records = []
    for config in configs:
        oof = np.zeros(len(dev))
        for fold in range(5):
            train, valid = dev[fold_ids != fold], dev[fold_ids == fold]
            fitted = make_estimator(config).fit(x.iloc[train], y[train])
            oof[fold_ids == fold] = fitted.predict_proba(x.iloc[valid])[:,1]
        record = dict(config=config, development_cv=metrics(y[dev],oof))
        records.append(record)
        print(json.dumps(record), flush=True)
        (out / "search.json").write_text(json.dumps(records,indent=2)+"\n")
    selected = min(records[:-1],key=lambda r:r["development_cv"]["log_loss"])
    incumbent = records[-1]
    # Correcting a committee's context-dependent slopes is distinct from
    # retaining applicants' contexts: scoring still averages common profiles.
    models = {"context_slopes": GrantModel(selected["config"]).fit(history),
              "context_slopes_blend": GrantEnsemble([baseline,selected["config"]],[.5,.5]).fit(history)}
    for name,model in models.items():
        joblib.dump(model,out/f"{name}.joblib",compress=3)
    applicants = pd.read_csv(ROOT / "data/candidats_evaluation.csv")
    baseline_model = GrantModel(baseline).fit(history)
    baseline_predictions = baseline_model.allocate(applicants)
    submissions = []
    for name in models:
        artifact = out/f"{name}.joblib"
        model = joblib.load(artifact)
        path = ROOT/"upload_model_round6"/f"{name}.csv"
        row = write_predictions(model, applicants, path)
        with tempfile.TemporaryDirectory() as temp:
            regenerated = Path(temp)/"predictions.csv"
            write_predictions(joblib.load(artifact), applicants, regenerated)
            assert regenerated.read_bytes() == path.read_bytes()
        sample = applicants.iloc[:12].copy()
        scores = model.score(sample)
        sample.id_candidat = [f"NEW{i}" for i in range(len(sample))]
        sample["decision_octroi"] = 1
        sample.region_administrative = "Montreal"
        sample.revenu_familial_estime = 100000.
        np.testing.assert_allclose(scores, model.score(sample), rtol=0, atol=1e-13)
        np.testing.assert_allclose(scores, model.score(sample.iloc[::-1])[::-1],rtol=0,atol=1e-13)
        np.testing.assert_allclose(scores[:1], model.score(sample.iloc[:1]),rtol=0,atol=1e-13)
        row.update(model=name, artifact=str(artifact.relative_to(ROOT)),artifact_sha256=digest(artifact),
                   different_decisions_from_incumbent=int((model.allocate(applicants)!=baseline_predictions).sum()),
                   verification_passed=True, reference_accuracy=None)
        submissions.append(row)
        print(json.dumps(row),flush=True)
    report = dict(search=records, selected=selected, incumbent=incumbent,
                  historical_log_loss_improved=selected["development_cv"]["log_loss"]<incumbent["development_cv"]["log_loss"],
                  training_sha256=digest(ROOT/"data/donnees_demandes.csv"),
                  reference_labels_used=False, historical_holdout_used=False,
                  ensemble_weights="Fixed equal weights; not optimized on preview results",
                  submissions=submissions)
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
