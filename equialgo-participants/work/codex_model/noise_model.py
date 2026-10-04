"""Select a historical label-noise model on the existing development folds."""
import json
from pathlib import Path
import tempfile

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import check_grad
from threadpoolctl import threadpool_limits

from model import GrantModel, NoisyLogistic, ROOT, digest, features, make_estimator, write_predictions
from train import metrics

HERE = Path(__file__).resolve().parent


def main():
    out = HERE / "noise_model"
    out.mkdir(exist_ok=True)
    rng = np.random.default_rng(190)
    x_test = rng.normal(size=(25,4))
    y_test = rng.integers(0,2,size=25)
    theta = rng.normal(size=5)
    estimator = NoisyLogistic(noise_floor=.08)
    error = check_grad(lambda t: estimator.objective(t,x_test,y_test)[0],
                       lambda t: estimator.objective(t,x_test,y_test)[1], theta)
    assert error < 1e-6, error
    history = pd.read_csv(ROOT / "data/donnees_demandes.csv")
    x, y = features(history), history.decision_octroi.to_numpy()
    split = pd.read_csv(HERE / "artifacts/validation_split.csv")
    assert np.array_equal(split.id_candidat, history.id_candidat)
    dev = np.flatnonzero(split.split.to_numpy() == "development")
    folds = split.cv_fold.to_numpy()[dev]
    records = []
    for floor in [0., .03, .06, .09]:
        config = dict(family="sparse_spline", knots=5, curve_hours=True, C=3., noise_floor=floor)
        oof = np.zeros(len(dev))
        for fold in range(5):
            train, valid = dev[folds != fold], dev[folds == fold]
            fitted = make_estimator(config).fit(x.iloc[train],y[train])
            oof[folds == fold] = fitted.predict_proba(x.iloc[valid])[:,1]
        record = dict(config=config, development_cv=metrics(y[dev],oof))
        records.append(record)
        print(json.dumps(record), flush=True)
    selected = min(records, key=lambda r: r["development_cv"]["log_loss"])
    model = GrantModel(selected["config"]).fit(history)
    artifact = out / "noise_aware.joblib"
    joblib.dump(model,artifact,compress=3)
    applicants = pd.read_csv(ROOT / "data/candidats_evaluation.csv")
    path = ROOT / "upload_model_round5/noise_aware.csv"
    row = write_predictions(joblib.load(artifact),applicants,path)
    with tempfile.TemporaryDirectory() as temp:
        generated = Path(temp) / "predictions.csv"
        write_predictions(joblib.load(artifact),applicants,generated)
        assert path.read_bytes() == generated.read_bytes()
    sample = applicants.iloc[:10].copy()
    scores = model.score(sample)
    sample.id_candidat = [f"NEW{i}" for i in range(len(sample))]
    sample["decision_octroi"] = 1
    np.testing.assert_allclose(scores, model.score(sample), rtol=0, atol=1e-13)
    np.testing.assert_allclose(scores[:1], model.score(sample.iloc[:1]), rtol=0, atol=1e-13)
    report = dict(search=records, selected=selected, submission=row,
                  gradient_check_error=float(error), verification_passed=True,
                  training_sha256=digest(ROOT / "data/donnees_demandes.csv"),
                  artifact_sha256=digest(artifact), reference_accuracy=None,
                  reference_labels_used=False, holdout_used_for_selection=False)
    (out / "report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(row),flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
