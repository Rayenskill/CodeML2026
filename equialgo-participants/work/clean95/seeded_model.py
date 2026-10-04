"""Seed-ensembled counterfactual merit model (clean: historical labels only).

Idea. The committee's decisions mix merit (academic score, working hours) with
nuisance signal (family income, region). We fit a sparse additive-spline logistic
model of the committee and then score every applicant *counterfactually*: their
own academic score and hours are kept, every other feature is replaced by a
common reference profile drawn from historical applicants and the predicted
probabilities are averaged. This is the corrected "merit" score.

Robustness. Everything random is controlled by an explicit seed and averaged out:
  * each member is fitted on its own bootstrap resample of the history,
  * each member neutralises over its own 256 reference profiles,
  * the final score is the mean over N_SEEDS members.
So no single bootstrap draw, knot placement or profile sample decides anyone.

No platform score, correction file or evaluation label is read at any point.

    python work/clean95/seeded_model.py validate   # repeated splits + seed stability
    python work/clean95/seeded_model.py predict    # writes upload_final/predictions.csv
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "work/codex_model"))
from model import MERIT, REMOTE, features, make_estimator  # noqa: E402

CONFIG = dict(family="sparse_spline", knots=5, curve_hours=True, C=3.0)
N_SEEDS = 30
N_PROFILES = 256
GRANT_RATE = 0.40
OUT = ROOT / "upload_final"


def fit_member(history, seed):
    rng = np.random.default_rng(seed)
    boot = history.iloc[rng.integers(0, len(history), len(history))]
    x = features(boot)
    estimator = make_estimator(CONFIG).fit(x, boot.decision_octroi.to_numpy())
    profiles = x.sample(n=N_PROFILES, random_state=seed).reset_index(drop=True)
    return estimator, profiles


def member_score(member, applicants_x):
    estimator, profiles = member
    total = np.zeros(len(applicants_x))
    merit = applicants_x[MERIT].to_numpy()
    for i in range(len(profiles)):
        common = profiles.iloc[np.full(len(applicants_x), i)].reset_index(drop=True)
        common[MERIT] = merit
        total += estimator.predict_proba(common)[:, 1]
    return total / len(profiles)


class SeededMerit:
    def __init__(self, seeds):
        self.seeds = list(seeds)

    def fit(self, history):
        with threadpool_limits(limits=1):
            self.members_ = Parallel(n_jobs=10)(delayed(fit_member)(history, s) for s in self.seeds)
        return self

    def member_scores(self, applicants):
        x = features(applicants)
        with threadpool_limits(limits=1):
            return np.array(Parallel(n_jobs=10)(delayed(member_score)(m, x) for m in self.members_))

    def score(self, applicants):
        return self.member_scores(applicants).mean(0)

    def allocate(self, applicants, scores=None):
        """Top GRANT_RATE of the cohort; ties broken by academic score, then hours."""
        x = features(applicants)
        scores = self.score(applicants) if scores is None else scores
        order = np.lexsort((-x.hours.to_numpy(), -x.academic.to_numpy(), -scores))
        out = np.zeros(len(x), dtype=int)
        out[order[:round(GRANT_RATE * len(x))]] = 1
        return out


def top(scores, share=GRANT_RATE):
    out = np.zeros(len(scores), dtype=int)
    out[np.argsort(-scores, kind="stable")[:round(share * len(scores))]] = 1
    return out


def validate():
    history = pd.read_csv(ROOT / "data/donnees_demandes.csv")
    applicants = pd.read_csv(ROOT / "data/candidats_evaluation.csv")
    report = {"config": CONFIG, "n_seeds": N_SEEDS, "n_profiles": N_PROFILES}

    # 1. Generalisation: 5 repeated 80/20 splits with different split seeds.
    splits = []
    for r in range(5):
        rng = np.random.default_rng(1000 + r)
        perm = rng.permutation(len(history))
        test, train = perm[:2000], perm[2000:]
        tr, te = history.iloc[train], history.iloc[test]
        model = SeededMerit(range(r * 100, r * 100 + 10)).fit(tr)
        member_x = features(te)
        # Committee-fidelity of the *uncorrected* members (does the fit generalise?)
        raw = np.mean([m[0].predict_proba(member_x)[:, 1] for m in model.members_], axis=0)
        y = te.decision_octroi.to_numpy()
        merit = model.score(te)
        remote = te.region_administrative.isin(REMOTE).to_numpy()
        pred = top(merit)
        splits.append({
            "split": r,
            "fit_committee_accuracy": float(((raw > .5) == y).mean()),
            "fit_committee_auc": float(roc_auc_score(y, raw)),
            "merit_top40_vs_committee_accuracy": float((pred == y).mean()),
            "merit_grant_rate_remote": float(pred[remote].mean()),
            "merit_grant_rate_centre": float(pred[~remote].mean()),
            "committee_grant_rate_remote": float(y[remote].mean()),
            "committee_grant_rate_centre": float(y[~remote].mean()),
        })
        print(json.dumps(splits[-1]), flush=True)
    report["repeated_splits"] = splits
    for k in ("fit_committee_accuracy", "fit_committee_auc", "merit_top40_vs_committee_accuracy"):
        v = np.array([s[k] for s in splits])
        report[k + "_mean_sd"] = [float(v.mean()), float(v.std(ddof=1))]

    # 2. Seed stability on the evaluation cohort (features only).
    model = SeededMerit(range(N_SEEDS)).fit(history)
    ms = model.member_scores(applicants)
    decisions = np.array([top(s) for s in ms])
    ens = top(ms.mean(0))
    flips = [(d != ens).sum() for d in decisions]
    unanimous = float(((decisions.sum(0) == 0) | (decisions.sum(0) == N_SEEDS)).mean())
    report["stability"] = {
        "single_seed_vs_ensemble_flips_mean": float(np.mean(flips)),
        "single_seed_vs_ensemble_flips_max": int(np.max(flips)),
        "unanimous_applicants_share": unanimous,
        "applicants_with_split_vote": int(((decisions.sum(0) > 0) & (decisions.sum(0) < N_SEEDS)).sum()),
    }
    # 3. Group parity of the final allocation on the evaluation cohort.
    remote = applicants.region_administrative.isin(REMOTE).to_numpy()
    report["eval_grant_rate"] = {"remote": float(ens[remote].mean()), "centre": float(ens[~remote].mean()),
                                 "overall": float(ens.mean())}
    best = pd.read_csv(ROOT / "upload_model_best/predictions.csv").set_index("id_candidat").decision_octroi
    report["differs_from_verified_single_spline"] = int((ens != best.reindex(applicants.id_candidat).to_numpy()).sum())
    (ROOT / "work/clean95/validation_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "repeated_splits"}, indent=2))


def predict():
    history = pd.read_csv(ROOT / "data/donnees_demandes.csv")
    applicants = pd.read_csv(ROOT / "data/candidats_evaluation.csv")
    model = SeededMerit(range(N_SEEDS)).fit(history)
    decision = model.allocate(applicants)
    assert 0.36 <= decision.mean() <= 0.44 and len(decision) == 4000
    OUT.mkdir(exist_ok=True)
    pd.DataFrame({"id_candidat": applicants.id_candidat, "decision_octroi": decision}).to_csv(
        OUT / "predictions.csv", index=False)
    print("grants", int(decision.sum()), "->", OUT / "predictions.csv")


if __name__ == "__main__":
    {"validate": validate, "predict": predict}[sys.argv[1]]()
