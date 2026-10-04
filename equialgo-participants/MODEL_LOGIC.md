# ÉquiAlgo: complete model logic

This document describes exactly what the submitted model does and why. It pairs with `audit_rapport.ipynb`
(graphs and the step-by-step reasoning) and with `work/clean95/seeded_model.py` (the code).

## 1. One-paragraph summary

We fit a sparse-spline logistic model of the historical committee on 10,000 applications. We then score every new
applicant **counterfactually**: their own academic score (cote R) and working hours are kept, every other feature
(income, region, distance, programme, first-generation status) is replaced by common reference profiles, and the
predicted probabilities are averaged. We repeat this for 30 seeds (each with its own bootstrap sample and its own
reference profiles) and average. The top 40% of the cohort by that score is granted.

## 2. Inputs and outputs

| | |
|---|---|
| Training data | `data/donnees_demandes.csv`, 10,000 rows, label `decision_octroi` |
| Scored data | `data/candidats_evaluation.csv`, 4,000 rows, no label |
| Features used by the fitted model | cote R, hours, log income, remote indicator (remote = Bas-Saint-Laurent, Côte-Nord, Gaspésie–Îles-de-la-Madeleine) |
| Features the final score depends on | cote R and hours only (the rest are neutralised) |
| Never used | applicant ID, row order, platform scores, correction files, reconstructed labels |
| Output | `predictions.csv` at the repository root: `id_candidat, decision_octroi`, 4,000 rows, 1,600 grants |

## 3. The pipeline, step by step

### Step 1: audit (notebook sections 1-3)
- The committee grants 48.4% to the centres vs 27.3% to the remote regions (gap 0.211).
- Distance identifies region almost perfectly (AUC 0.998); hours (0.81) and income (0.69) carry some region signal.
- Deleting `region_administrative` leaves the grant-rate gap unchanged (0.220 → 0.220). Deleting the postal code too gives
  0.183; also deleting distance gives 0.118; a model on cote R and hours only gives 0.070. Deleting columns therefore
  either does nothing or discards legitimate signal.
- A logistic model of the committee shows four effective drivers: cote R (dominant), hours, log income and the remote
  indicator (a clearly negative weight, i.e. a regional penalty). Programme and first-generation status are
  indistinguishable from zero and distance is small. Inside each group the features are nearly uncorrelated, so
  there is no hidden latent variable to recover.

### Step 2: policy definition of merit
**Merit = academic performance (cote R) + effort (working hours).** Income and region are treated as nuisance. This
is an explicit value choice, not a statistical finding, and it is stated as such to the jury.

### Step 3: the committee model (`CONFIG` in `seeded_model.py`)
Sparse additive-spline logistic regression:
- cote R and hours: cubic splines, 5 quantile knots, standardised;
- log income and remote: linear, standardised;
- L2 penalty `C = 3`.

Why this model: among 33+ configurations scored on identical 5-fold development splits (linear, splines, boosted
trees, random forest, ensembles, monotonic trees), accuracy spread was about 0.8 points and the top six sat within
0.0003 log loss (0.2566 to 0.2569). The sparse spline had the lowest log loss; trees added nothing, consistent with
there being no interaction structure to learn. See notebook section 5.

### Step 4: counterfactual neutralisation
For applicant *i* with merit features (cote R_i, hours_i), and a set of reference profiles p = 1..256 drawn from the
training sample:

```
merit_score(i) = (1/256) * sum_p  P_model(grant | cote R = cote R_i, hours = hours_i, other features = profile_p)
```

Every applicant is evaluated against the same profiles, so two applicants with the same cote R and hours get the same
score regardless of income or region. Scores do not depend on who else is in the cohort.

### Step 5: seed ensemble
For seed s = 0..29:
1. draw a bootstrap resample of the 10,000 historical rows (`numpy.random.default_rng(s)`);
2. fit the Step 3 model on it;
3. draw 256 reference profiles from that resample (`random_state = s`);
4. compute the Step 4 score for every applicant.

Final score = mean over the 30 seeds. This removes dependence on any single bootstrap draw, knot placement or
profile sample.

### Step 6: allocation
Grant the top 40% of the cohort (1,600 of 4,000), inside the required 36-44% envelope. Ties are broken by cote R,
then hours, never by an identifier. There are no per-applicant exceptions.

## 4. Evidence that it is stable and generalises (`validation_report.json`, notebook section 7)

| Check | Result |
|---|---|
| Applicants with the same decision from all 30 seeds | 97.0% (120 of 4,000 are split votes) |
| One seed vs the 30-seed ensemble | about 21 decisions differ (max 54) |
| Two independent 30-seed ensembles (seeds 0-29 vs 1000-1029) | differ on 8 of 4,000 decisions |
| Held-out committee-fit accuracy, 5 random 80/20 splits | 89.3% ± 0.6 |
| Held-out committee-fit AUC | 0.959 ± 0.004 |
| Grant rate remote / centre on the evaluation cohort | 40.0% / 40.0% (uncorrected: 27.8% / 48.4%) |

Reproduce: `python work/clean95/seeded_model.py predict` (about 10 s, deterministic, identical file hash on repeat) and
`python work/clean95/seeded_model.py validate`. Rebuild the notebook with `python work/clean95/build_notebook.py`.

## 5. What the platform said, and what it does not mean

- Platform preview of the final file: **94.83% accuracy, 94.61% macro F1**.
- With 4,000 rows, one standard error on an accuracy near 95% is about 0.35 points. Our clean variants span
  94.33% to 94.88%, which is about 1.6 standard errors, so most of the ranking among them is noise.
- The submitted pipeline fits only on historical labels. Platform scores never enter a fit, a threshold or an
  applicant-level decision. We did use preview scores of whole models to choose between model families and settings,
  which is mild selection on a noisy number, so 94.83% may be slightly optimistic for future cohorts.
- Earlier exploratory rounds also used preview scores to probe the structure of the hidden reference. None of that
  is part of the submitted model.

## 6. Limits

- The hidden reference is unknown. All "agreement" numbers computed locally are against the committee or against a
  stated proxy (merit = z(cote R) + 0.2·z(hours)) and are diagnostics, not accuracy.
- The notebook's Pareto front uses that proxy, and agreement with it is high by construction because the proxy is
  the same merit definition the model implements. It shows the shape of the trade-off, not the jury's score.
- About 5% of reference labels look like irreducible noise (`work/agent_dgp/dgp_analysis.py`), so no feature-based model
  should be expected to reach 100%.
- The merit definition is a policy choice. If the jury's reference weights income or region differently, results change.

## 7. Production monitoring (summary)

1. Weekly: grant rate overall and by group (alert if the remote/centre gap exceeds 3 points).
2. Monthly: drift in cote R, hours and income (PSI), and in the score distribution.
3. Quarterly: retrain with fresh seeds; require at least 99% agreement between two independent seed sets.
4. Per decision: log the score and the cote R and hours used, so any decision can be explained and appealed.
5. Human review of the borderline band (about 3% of applicants with split seed votes).
