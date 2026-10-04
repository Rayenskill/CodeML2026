"""Validate alternate probability links without reading evaluation outcomes."""
import json
from pathlib import Path
import tempfile

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import check_grad
from threadpoolctl import threadpool_limits

from model import GrantModel, ProbitClassifier, ROOT, digest, features, make_estimator, write_predictions
from train import metrics

HERE = Path(__file__).resolve().parent


def main():
    out = HERE/"probit"
    out.mkdir(exist_ok=True)
    rng=np.random.default_rng(402)
    toy_x=rng.normal(size=(30,5));toy_y=rng.integers(0,2,size=30);theta=rng.normal(size=6)
    learner=ProbitClassifier()
    error=check_grad(lambda t:learner.objective(t,toy_x,toy_y)[0],
                     lambda t:learner.objective(t,toy_x,toy_y)[1],theta)
    assert error<1e-6,error
    # Stable likelihood and gradient even in the probability tails.
    extreme=np.ones((2,5));extreme[1]=-1
    loss,gradient=learner.objective(np.r_[np.full(5,10.),0.],extreme,np.array([0.,1.]))
    assert np.isfinite(loss) and np.isfinite(gradient).all()
    history=pd.read_csv(ROOT/"data/donnees_demandes.csv")
    x,y=features(history),history.decision_octroi.to_numpy()
    split=pd.read_csv(HERE/"artifacts/validation_split.csv")
    assert np.array_equal(split.id_candidat,history.id_candidat)
    dev=np.flatnonzero(split.split.to_numpy()=="development")
    fold_ids=split.cv_fold.to_numpy()[dev]
    baseline=dict(family="sparse_spline",knots=5,curve_hours=True,C=3.)
    configs=[dict(family="linear",reduced=True,C=c,link="probit") for c in [.1,3.,30.]]
    configs += [dict(baseline,C=c,link="probit") for c in [.1,3.,30.]]
    configs += [baseline,dict(family="linear",reduced=True,C=3.)]
    records=[]
    for config in configs:
        oof=np.zeros(len(dev))
        for fold in range(5):
            train,valid=dev[fold_ids!=fold],dev[fold_ids==fold]
            fitted=make_estimator(config).fit(x.iloc[train],y[train])
            oof[fold_ids==fold]=fitted.predict_proba(x.iloc[valid])[:,1]
        row=dict(config=config,development_cv=metrics(y[dev],oof))
        records.append(row)
        print(json.dumps(row),flush=True)
        (out/"search.json").write_text(json.dumps(records,indent=2)+"\n")
    chosen=min(records[:6],key=lambda r:r["development_cv"]["log_loss"])
    report=dict(search=records,selected=chosen,gradient_check_error=float(error),
                training_sha256=digest(ROOT/"data/donnees_demandes.csv"),
                reference_labels_used=False,historical_holdout_used=False,submissions=[])
    # Do not recommend another preview unless this model improves on the
    # incumbent's historical development likelihood.
    improved=chosen["development_cv"]["log_loss"]<records[6]["development_cv"]["log_loss"]
    report["historical_log_loss_improved"]=improved
    if improved:
        model=GrantModel(chosen["config"]).fit(history)
        artifact=out/"probit.joblib"
        joblib.dump(model,artifact,compress=3)
        applicants=pd.read_csv(ROOT/"data/candidats_evaluation.csv")
        path=ROOT/"upload_model_round7/probit.csv"
        row=write_predictions(joblib.load(artifact),applicants,path)
        with tempfile.TemporaryDirectory() as temp:
            regenerated=Path(temp)/"predictions.csv"
            write_predictions(joblib.load(artifact),applicants,regenerated)
            assert regenerated.read_bytes()==path.read_bytes()
        sample=applicants.iloc[:12].copy();scores=model.score(sample)
        sample.id_candidat=[f"NEW{i}" for i in range(len(sample))]
        sample["decision_octroi"]=1
        sample.region_administrative="Montreal";sample.revenu_familial_estime=100000.
        np.testing.assert_allclose(scores,model.score(sample),rtol=0,atol=1e-13)
        np.testing.assert_allclose(scores,model.score(sample.iloc[::-1])[::-1],rtol=0,atol=1e-13)
        np.testing.assert_allclose(scores[:1],model.score(sample.iloc[:1]),rtol=0,atol=1e-13)
        incumbent=GrantModel(baseline).fit(history)
        row.update(artifact=str(artifact.relative_to(ROOT)),artifact_sha256=digest(artifact),
                   different_decisions_from_incumbent=int((model.allocate(applicants)!=incumbent.allocate(applicants)).sum()),
                   verification_passed=True,reference_accuracy=None)
        report["submissions"].append(row)
        print(json.dumps(row),flush=True)
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")


if __name__=="__main__":
    with threadpool_limits(limits=2):
        main()
