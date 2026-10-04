"""Trainable grant model; fits historical labels and scores unseen applications.

The fitted committee model controls for nuisance features. The corrected score
averages its predicted probability over common, training-only nuisance profiles,
holding the applicant's academic score and working hours fixed. Treating these
two features as legitimate is an explicit mitigation assumption, not a claim
that the hidden reference labels have been recovered.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, SplineTransformer, StandardScaler
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[2]
REMOTE = {"Bas-Saint-Laurent", "Cote-Nord", "Gaspesie-Iles-de-la-Madeleine"}
REGIONS = REMOTE | {"Montreal", "Capitale-Nationale"}
MERIT = ["academic", "hours"]
NUMERIC = MERIT + ["log_income", "log_distance", "first_generation", "remote"]
CATEGORICAL = ["programme", "region", "postal"]
REQUIRED = [
    "cote_r_equivalent", "heures_travail_semaine", "revenu_familial_estime",
    "distance_domicile_campus_km", "premiere_generation_universitaire",
    "programme_etudes", "region_administrative", "code_postal_3",
]
SEED = 1042026


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def features(frame: pd.DataFrame) -> pd.DataFrame:
    """Whitelist features: identifiers and any supplied labels are never inputs."""
    missing = set(REQUIRED) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing application fields: {sorted(missing)}")
    if frame[REQUIRED].isna().any().any():
        raise ValueError("Missing application values must be resolved before scoring.")
    if not set(frame.region_administrative.unique()) <= REGIONS:
        raise ValueError("The model was designed for the five challenge regions.")
    raw = frame[REQUIRED[:5]].to_numpy(dtype=float)
    if not np.isfinite(raw).all():
        raise ValueError("Numeric fields must be finite.")
    if (frame.revenu_familial_estime <= 0).any():
        raise ValueError("Income must be positive.")
    if (frame.distance_domicile_campus_km < 0).any() or (frame.heures_travail_semaine < 0).any():
        raise ValueError("Distance and working hours cannot be negative.")
    if not frame.premiere_generation_universitaire.isin([0, 1]).all():
        raise ValueError("First-generation status must be 0 or 1.")
    return pd.DataFrame({
        "academic": frame.cote_r_equivalent.to_numpy(dtype=float),
        "hours": frame.heures_travail_semaine.to_numpy(dtype=float),
        "log_income": np.log(frame.revenu_familial_estime.to_numpy(dtype=float)),
        "log_distance": np.log1p(frame.distance_domicile_campus_km.to_numpy(dtype=float)),
        "first_generation": frame.premiere_generation_universitaire.to_numpy(dtype=float),
        "remote": frame.region_administrative.isin(REMOTE).to_numpy(dtype=float),
        "merit_group": frame.region_administrative.isin(REMOTE).to_numpy(dtype=float),
        "programme": frame.programme_etudes.astype(str).to_numpy(),
        "region": frame.region_administrative.astype(str).to_numpy(),
        "postal": frame.code_postal_3.astype(str).to_numpy(),
    }, index=frame.index)


class TensorMeritBasis(TransformerMixin, BaseEstimator):
    """Smooth main effects and tensor interactions for academic score/work hours."""

    def __init__(self, knots=4):
        self.knots = knots

    def fit(self, x, y=None):
        self.spline_ = SplineTransformer(n_knots=self.knots, degree=3,
            knots="quantile", extrapolation="linear", include_bias=False).fit(x)
        self.width_ = self.spline_.transform(x[:1]).shape[1] // 2
        self.n_features_in_ = 2
        return self

    def transform(self, x):
        basis = self.spline_.transform(x)
        a, b = basis[:, :self.width_], basis[:, self.width_:]
        return np.column_stack([basis, np.einsum("ni,nj->nij", a, b).reshape(len(x), -1)])

    def get_feature_names_out(self, input_features=None):
        return np.array([f"academic_{i}" for i in range(self.width_)] +
            [f"hours_{i}" for i in range(self.width_)] +
            [f"academic_hours_{i}_{j}" for i in range(self.width_) for j in range(self.width_)], dtype=object)


class RegionalMeritNormalizer(TransformerMixin, BaseEstimator):
    """Training-fitted within-group merit centering or standardization."""

    def __init__(self, standardize=True):
        self.standardize = standardize

    def fit(self, x, y=None):
        data = np.asarray(x, dtype=float)
        self.means_, self.scales_ = {}, {}
        for group in [0, 1]:
            values = data[data[:, 2] == group, :2]
            if len(values) < 2:
                raise ValueError("Both regional groups need training examples.")
            self.means_[group] = values.mean(axis=0)
            self.scales_[group] = np.maximum(values.std(axis=0), 1e-8) if self.standardize else np.ones(2)
        self.n_features_in_ = 3
        return self

    def transform(self, x):
        data = np.asarray(x, dtype=float)
        if not np.isin(data[:, 2], [0, 1]).all():
            raise ValueError("Merit group must be binary.")
        out = np.empty((len(data), 2))
        for group in [0, 1]:
            mask = data[:, 2] == group
            out[mask] = (data[mask, :2] - self.means_[group]) / self.scales_[group]
        return out

    def get_feature_names_out(self, input_features=None):
        return np.array(["regional_academic", "regional_hours"], dtype=object)


class ProbitClassifier(BaseEstimator):
    """Penalized Bernoulli likelihood with a normal-CDF probability link."""

    def __init__(self, C=3.):
        self.C = C

    def objective(self, theta, x, y):
        z = x @ theta[:-1] + theta[-1]
        log_p, log_q, log_density = norm.logcdf(z), norm.logsf(z), norm.logpdf(z)
        penalty = 1/(self.C*len(y))
        loss = -np.mean(y*log_p+(1-y)*log_q)+.5*penalty*np.dot(theta[:-1],theta[:-1])
        residual = (-y*np.exp(log_density-log_p)+(1-y)*np.exp(log_density-log_q))/len(y)
        gradient = np.r_[x.T @ residual + penalty*theta[:-1],residual.sum()]
        return loss,gradient

    def fit(self, x, y):
        if self.C <= 0:
            raise ValueError("C must be positive.")
        x,y = np.asarray(x,dtype=float),np.asarray(y,dtype=float)
        initial = LogisticRegression(C=self.C,max_iter=3000).fit(x,y)
        theta = np.r_[initial.coef_[0],initial.intercept_[0]] / 1.6
        result = minimize(self.objective,theta,args=(x,y),jac=True,method="L-BFGS-B",
                          options=dict(maxiter=3000,ftol=1e-12,gtol=1e-7))
        if not result.success:
            raise RuntimeError(f"Probit fit failed: {result.message}")
        self.coef_,self.intercept_ = result.x[:-1][None,:],result.x[-1:]
        self.classes_ = np.array([0,1])
        self.n_features_in_ = x.shape[1]
        return self

    def predict_proba(self,x):
        p = norm.cdf(np.asarray(x) @ self.coef_[0] + self.intercept_[0])
        return np.c_[1-p,p]


class ContextMeritBasis(TransformerMixin, BaseEstimator):
    """Training-fitted merit splines with context-dependent slopes."""

    def __init__(self, knots=5):
        self.knots = knots

    def fit(self, x, y=None):
        data = np.asarray(x, dtype=float)
        self.spline_ = SplineTransformer(n_knots=self.knots, degree=3,
            knots="quantile", extrapolation="linear", include_bias=False).fit(data[:, :2])
        self.context_mean_ = data[:, 2:].mean(axis=0)
        self.context_scale_ = np.maximum(data[:, 2:].std(axis=0), 1e-8)
        self.width_ = self.spline_.transform(data[:1, :2]).shape[1]
        self.n_features_in_ = data.shape[1]
        return self

    def transform(self, x):
        data = np.asarray(x, dtype=float)
        basis = self.spline_.transform(data[:, :2])
        context = (data[:, 2:] - self.context_mean_) / self.context_scale_
        products = np.einsum("ni,nj->nij", basis, context).reshape(len(data), -1)
        return np.column_stack([basis, products])

    def get_feature_names_out(self, input_features=None):
        return np.array([f"merit_{i}" for i in range(self.width_)] +
            [f"merit_{i}_context_{j}" for i in range(self.width_)
             for j in range(self.n_features_in_-2)], dtype=object)


class NoisyLogistic(BaseEstimator):
    """Bernoulli model with a fixed symmetric historical-label flip floor."""

    def __init__(self, C=3., noise_floor=.05):
        self.C = C
        self.noise_floor = noise_floor

    def objective(self, theta, x, y):
        latent = expit(x @ theta[:-1] + theta[-1])
        p = np.clip(self.noise_floor + (1-2*self.noise_floor)*latent, 1e-12, 1-1e-12)
        loss = -np.mean(y*np.log(p)+(1-y)*np.log1p(-p))
        penalty = 1/(self.C*len(y))
        loss += .5*penalty*np.dot(theta[:-1], theta[:-1])
        residual = (p-y)/(p*(1-p))*(1-2*self.noise_floor)*latent*(1-latent)/len(y)
        gradient = np.r_[x.T @ residual + penalty*theta[:-1], residual.sum()]
        return loss, gradient

    def fit(self, x, y):
        if not 0 <= self.noise_floor < .5 or self.C <= 0:
            raise ValueError("Invalid noise floor or regularization.")
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        initial = LogisticRegression(C=self.C, max_iter=3000).fit(x, y)
        theta = np.r_[initial.coef_[0], initial.intercept_[0]]
        result = minimize(self.objective, theta, args=(x,y), jac=True, method="L-BFGS-B",
                          options=dict(maxiter=3000, ftol=1e-12, gtol=1e-7))
        if not result.success:
            raise RuntimeError(f"Noise-aware fit failed: {result.message}")
        self.coef_ = result.x[:-1][None, :]
        self.intercept_ = result.x[-1:]
        self.classes_ = np.array([0,1])
        self.n_features_in_ = x.shape[1]
        return self

    def predict_proba(self, x):
        latent = expit(np.asarray(x) @ self.coef_[0] + self.intercept_[0])
        p = self.noise_floor + (1-2*self.noise_floor)*latent
        return np.c_[1-p,p]


def make_estimator(config: dict) -> Pipeline:
    family = config["family"]
    categorical = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    if family == "linear":
        if config.get("reduced", False):
            pre = ColumnTransformer([
                ("numeric", StandardScaler(), MERIT + ["log_income", "remote"])
            ])
        else:
            pre = ColumnTransformer([
                ("numeric", StandardScaler(), NUMERIC),
                ("categorical", categorical, CATEGORICAL),
            ])
        learner = LogisticRegression(C=config["C"], max_iter=3000, random_state=SEED)
    elif family in {"tensor", "quadratic", "regional_spline", "context_spline"}:
        if family == "tensor":
            merit = Pipeline([("basis", TensorMeritBasis(config["knots"])), ("scale", StandardScaler())])
            merit_columns = MERIT
        elif family == "quadratic":
            merit = Pipeline([("scale_input", StandardScaler()),
                              ("polynomial", PolynomialFeatures(degree=2, include_bias=False)),
                              ("scale_output", StandardScaler())])
            merit_columns = MERIT
        else:
            steps = []
            merit_columns = MERIT
            if family == "regional_spline":
                steps.append(("regional", RegionalMeritNormalizer(config["standardize"])))
                merit_columns = MERIT + ["merit_group"]
            steps.extend([("spline", SplineTransformer(n_knots=5, degree=3, knots="quantile",
                extrapolation="linear", include_bias=False)), ("scale", StandardScaler())])
            merit = Pipeline(steps)
        controls = ["log_income", "remote"]
        transformers = [("merit", merit, merit_columns)]
        if family == "context_spline":
            controls += ["first_generation", "log_distance"]
            transformers.append(("programme", categorical, ["programme"]))
        transformers.append(("controls", StandardScaler(), controls))
        pre = ColumnTransformer(transformers)
        learner = LogisticRegression(C=config["C"], max_iter=4000, random_state=SEED)
    elif family == "context_slopes":
        context = config["context"]
        pre = ColumnTransformer([
            ("merit_context", Pipeline([("basis", ContextMeritBasis()),
                ("scale", StandardScaler())]), MERIT + context),
            ("controls", StandardScaler(), ["log_income", "remote"]),
        ])
        learner = LogisticRegression(C=config["C"], max_iter=4000, random_state=SEED)
    elif family == "sparse_spline":
        curved = MERIT if config["curve_hours"] else ["academic"]
        pre = ColumnTransformer([
            ("curves", Pipeline([
                ("spline", SplineTransformer(n_knots=config["knots"], degree=config.get("degree", 3),
                    knots=config.get("knot_spacing", "quantile"), extrapolation="linear", include_bias=False)),
                ("scale", StandardScaler()),
            ]), curved),
            ("controls", StandardScaler(),
             [v for v in MERIT + ["log_income", "remote"] if v not in curved]),
        ])
        learner = LogisticRegression(C=config["C"], max_iter=3000, random_state=SEED)
        if "noise_floor" in config:
            learner = NoisyLogistic(C=config["C"], noise_floor=config["noise_floor"])
    elif family == "monotonic":
        pre = ColumnTransformer([("numeric", "passthrough", MERIT + ["log_income", "remote"])])
        learner = HistGradientBoostingClassifier(
            max_iter=config["iterations"], max_leaf_nodes=config["leaves"],
            min_samples_leaf=80, l2_regularization=config["l2"],
            learning_rate=0.05, early_stopping=False, random_state=SEED,
            monotonic_cst=[1, 1, 1, 0],
        )
    elif family == "spline":
        curved = MERIT + (["log_income", "log_distance"] if config["all_splines"] else [])
        pre = ColumnTransformer([
            ("curves", Pipeline([
                ("spline", SplineTransformer(n_knots=config["knots"], degree=3,
                    knots="quantile", extrapolation="linear", include_bias=False)),
                ("scale", StandardScaler()),
            ]), curved),
            ("numeric", StandardScaler(), [v for v in NUMERIC if v not in curved]),
            ("categorical", categorical, ["programme", "region"]),
        ])
        learner = LogisticRegression(C=config["C"], max_iter=3000, random_state=SEED)
    else:
        pre = ColumnTransformer([
            ("numeric", "passthrough", NUMERIC),
            ("categorical", categorical, ["programme", "region"]),
        ])
        if family == "boosted":
            learner = HistGradientBoostingClassifier(
                max_iter=config["iterations"], max_leaf_nodes=config["leaves"],
                min_samples_leaf=config["min_leaf"], l2_regularization=config["l2"],
                learning_rate=0.05, early_stopping=False, random_state=SEED,
            )
        elif family == "forest":
            learner = RandomForestClassifier(n_estimators=300, min_samples_leaf=20,
                                             n_jobs=2, random_state=42)
        else:
            raise ValueError(f"Unknown family: {family}")
    if config.get("link") == "probit":
        learner = ProbitClassifier(C=config["C"])
    return Pipeline([("preprocess", pre), ("learner", learner)])


class GrantModel:
    """A persisted estimator that can score any new feature-complete applicant.

    predict() uses a threshold learned from historical features. allocate() is
    the separate, deterministic cohort budget policy (40% by default). Neither
    method stores labels or decisions for evaluation applicants.
    """

    def __init__(self, config: dict, grant_rate: float = 0.40):
        if not 0.36 <= grant_rate <= 0.44:
            raise ValueError("grant_rate must lie within the challenge's 36–44% envelope")
        self.config = dict(config)
        self.grant_rate = float(grant_rate)

    def fit(self, history: pd.DataFrame) -> "GrantModel":
        if "decision_octroi" not in history or not history.decision_octroi.isin([0, 1]).all():
            raise ValueError("Training requires binary historical decision_octroi labels.")
        x = features(history)
        self.estimator_ = make_estimator(self.config).fit(x, history.decision_octroi)
        # The same 64 training profiles are used for all model families so
        # counterfactual probabilities are comparable in the ensemble.
        count = 64
        self.profiles_ = x.sample(n=min(count, len(x)), random_state=SEED).reset_index(drop=True)
        self.threshold_ = float(np.quantile(self.score(history), 1 - self.grant_rate))
        self.training_rows_ = len(history)
        return self

    def committee_probability(self, applicants: pd.DataFrame) -> np.ndarray:
        return self.estimator_.predict_proba(features(applicants))[:, 1]

    def score(self, applicants: pd.DataFrame) -> np.ndarray:
        x = features(applicants)
        result = np.zeros(len(x))
        if not len(x):
            return result
        for i in range(len(self.profiles_)):
            common = self.profiles_.iloc[np.full(len(x), i)].reset_index(drop=True)
            common[MERIT] = x[MERIT].to_numpy()
            result += self.estimator_.predict_proba(common)[:, 1]
        return result / len(self.profiles_)

    def predict(self, applicants: pd.DataFrame) -> np.ndarray:
        return (self.score(applicants) >= self.threshold_).astype(np.int64)

    def allocate(self, applicants: pd.DataFrame, scores: np.ndarray | None = None) -> np.ndarray:
        """Allocate a fixed share of a cohort by model score, with feature-only ties."""
        x = features(applicants)
        scores = self.score(applicants) if scores is None else np.asarray(scores)
        if scores.shape != (len(x),) or not np.isfinite(scores).all():
            raise ValueError("One finite score per applicant is required.")
        if getattr(self, "allocation_rule", "cohort_budget") == "training_threshold":
            return (scores >= self.threshold_).astype(np.int64)
        # No identifier enters even the tie-break: academic score and working
        # hours resolve ties. Exact duplicate merit profiles are exchangeable;
        # stable input order is the final tie-break at the budget boundary.
        order = np.lexsort((-x.hours.to_numpy(), -x.academic.to_numpy(), -scores))
        result = np.zeros(len(x), dtype=np.int64)
        result[order[:round(self.grant_rate * len(x))]] = 1
        return result


class ContextGrantModel(GrantModel):
    """Feature-based correction with explicit retained context and strength.

    Retention values are mitigation hyperparameters. The underlying feature
    effects remain fitted to original historical labels. No decision lookup
    table or evaluation labels enter the score.
    """

    def __init__(self, config, retained_columns=(), remote_retention=0.0, income_retention=0.0, grant_rate=0.40):
        super().__init__(config, grant_rate)
        self.retained_columns = tuple(retained_columns)
        if not -0.25 <= remote_retention <= 1 or not -0.25 <= income_retention <= 1:
            raise ValueError("Context retention is outside the modeled range.")
        self.remote_retention = float(remote_retention)
        self.income_retention = float(income_retention)

    def score(self, applicants):
        x = features(applicants)
        result = np.zeros(len(x))
        if not len(x):
            return result
        for i in range(len(self.profiles_)):
            common = self.profiles_.iloc[np.full(len(x), i)].reset_index(drop=True)
            keep = MERIT + list(self.retained_columns)
            for column in keep:
                common[column] = x[column].to_numpy()
            common["remote"] = (1 - self.remote_retention) * common.remote.to_numpy() + self.remote_retention * x.remote.to_numpy()
            common["log_income"] = (1 - self.income_retention) * common.log_income.to_numpy() + self.income_retention * x.log_income.to_numpy()
            result += self.estimator_.predict_proba(common)[:, 1]
        return result / len(self.profiles_)


class GrantEnsemble(GrantModel):
    """Probability mixture with weights fitted from development OOF predictions."""

    def __init__(self, configs: list[dict], weights: list[float], grant_rate: float = 0.40):
        super().__init__({"family": "ensemble"}, grant_rate)
        self.configs = configs
        self.weights = np.asarray(weights, dtype=float)
        if self.weights.shape != (len(configs),) or not np.isfinite(self.weights).all():
            raise ValueError("One finite ensemble weight per model is required.")
        if (self.weights < 0).any() or not np.isclose(self.weights.sum(), 1):
            raise ValueError("Ensemble weights must be nonnegative and sum to one.")

    def fit(self, history: pd.DataFrame) -> "GrantEnsemble":
        self.models_ = [GrantModel(config, self.grant_rate).fit(history) for config in self.configs]
        self.threshold_ = float(np.quantile(self.score(history), 1 - self.grant_rate))
        self.training_rows_ = len(history)
        return self

    def committee_probability(self, applicants: pd.DataFrame) -> np.ndarray:
        return sum(weight * model.committee_probability(applicants)
                   for weight, model in zip(self.weights, self.models_))

    def score(self, applicants: pd.DataFrame) -> np.ndarray:
        return sum(weight * model.score(applicants)
                   for weight, model in zip(self.weights, self.models_))


class BaggedGrantModel(GrantEnsemble):
    """Average models fitted on bootstrap resamples of historical applications."""

    def __init__(self, config: dict, n_estimators: int = 20, grant_rate: float = 0.40):
        if n_estimators < 2:
            raise ValueError("Bagging requires at least two estimators.")
        super().__init__([config] * n_estimators, [1 / n_estimators] * n_estimators, grant_rate)
        self.config = {"family": "bagging", "base": config, "n_estimators": n_estimators}

    def fit(self, history: pd.DataFrame) -> "BaggedGrantModel":
        rng = np.random.default_rng(SEED + 8)
        self.models_ = []
        for config in self.configs:
            sample = rng.integers(0, len(history), size=len(history))
            self.models_.append(GrantModel(config, self.grant_rate).fit(history.iloc[sample]))
        self.threshold_ = float(np.quantile(self.score(history), 1 - self.grant_rate))
        self.training_rows_ = len(history)
        return self


def write_predictions(model: GrantModel, applicants: pd.DataFrame, output: Path) -> dict:
    if "id_candidat" not in applicants or not applicants.id_candidat.is_unique or applicants.id_candidat.isna().any():
        raise ValueError("The output requires a unique, nonmissing id_candidat for every row.")
    if not len(applicants):
        raise ValueError("The applicant cohort is empty.")
    scores = model.score(applicants)
    decision = model.allocate(applicants, scores)
    rate = float(decision.mean())
    if not 0.36 <= rate <= 0.44:
        raise ValueError("This cohort size cannot satisfy the budget with the requested rounded allocation.")
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id_candidat": applicants.id_candidat.to_numpy(),
                  "decision_octroi": decision}).to_csv(output, index=False)
    return {"file": str(output), "rows": len(applicants), "grants": int(decision.sum()),
            "grant_rate": rate, "sha256": digest(output),
            "allocation_rule": getattr(model, "allocation_rule", "cohort_budget"),
            "fixed_training_threshold_grant_rate": float(model.predict(applicants).mean())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=None,
                        help="Saved model; default is the best platform-verified clean model when available.")
    parser.add_argument("--input", type=Path, default=ROOT / "data/candidats_evaluation.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "upload_model_best/predictions.csv")
    args = parser.parse_args()
    if args.model is None:
        selection = Path(__file__).parent / "best_model.json"
        if selection.exists():
            best = json.loads(selection.read_text())
            args.model = ROOT / best["model_file"]
            if digest(args.model) != best["model_sha256"]:
                raise ValueError("The verified model artifact has changed; revalidate it before prediction.")
        else:
            args.model = Path(__file__).parent / "artifacts/model.joblib"
    # Only load a model artifact produced locally by train.py: joblib is pickle.
    model = joblib.load(args.model)
    print(json.dumps(write_predictions(model, pd.read_csv(args.input), args.output), indent=2))


if __name__ == "__main__":
    main()
