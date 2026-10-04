"""EquiAlgo mitigation: seed-ensembled counterfactual merit model -> predictions.csv + Pareto front.

Fits only on data/donnees_demandes.csv. No platform score, correction file or evaluation label is used.
Full logic: MODEL_LOGIC.md. Step-by-step analysis and graphs: audit_rapport.ipynb.

Run from equialgo-participants/:  python model_corrige.py
Writes predictions.csv at the repository root (4,000 rows, 1,600 grants) and pareto_front.png here.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "work/clean95"))
from seeded_model import N_SEEDS, SeededMerit, top  # noqa: E402
from model import features  # noqa: E402  (work/codex_model, added to the path by seeded_model)

REMOTE = {"Bas-Saint-Laurent", "Cote-Nord", "Gaspesie-Iles-de-la-Madeleine"}


def main():
    history = pd.read_csv(ROOT / "data/donnees_demandes.csv")
    applicants = pd.read_csv(ROOT / "data/candidats_evaluation.csv")
    model = SeededMerit(range(N_SEEDS)).fit(history)

    # 1. Decisions: counterfactual merit score, top 40% of the cohort.
    members = model.member_scores(applicants)
    neutral = members.mean(0)
    decision = model.allocate(applicants, neutral)
    assert len(decision) == 4000 and 0.36 <= decision.mean() <= 0.44
    pd.DataFrame({"id_candidat": applicants.id_candidat, "decision_octroi": decision}).to_csv(
        ROOT.parent / "predictions.csv", index=False)  # repository root, as the rules require
    print("predictions.csv written:", int(decision.sum()), "grants")

    # 2. Pareto front over correction strength (lambda) and budget, against a stated PROXY reference
    #    (top 40% by z(cote R) + 0.2 z(hours)). Agreement with it is high by construction; it shows the
    #    shape of the fairness/utility trade-off and is NOT an estimate of the jury's reference.
    raw = np.mean([m[0].predict_proba(features(applicants))[:, 1] for m in model.members_], axis=0)
    remote = applicants.region_administrative.isin(REMOTE).to_numpy()
    z = lambda c: (applicants[c] - history[c].mean()) / history[c].std()
    proxy = top((z("cote_r_equivalent") + 0.2 * z("heures_travail_semaine")).to_numpy())

    def metrics(pred):
        pos = proxy == 1
        tpr = pd.Series(pred[pos]).groupby(remote[pos]).mean()
        return (pred == proxy).mean() * 100, abs(tpr[False] - tpr[True])

    fig, ax = plt.subplots(figsize=(8, 5.2))
    for share, color in [(.36, "#718096"), (.40, "#2b6cb0"), (.44, "#2f855a")]:
        pts = [metrics(top(l * neutral + (1 - l) * raw, share)) for l in np.linspace(0, 1, 11)]
        gap, agree = [p[1] for p in pts], [p[0] for p in pts]
        ax.plot(gap, agree, "o-", color=color, label=f"budget {share:.0%}")
        ax.annotate("committee-style", (gap[0], agree[0]), fontsize=7, xytext=(4, 4), textcoords="offset points")
        ax.annotate("ours", (gap[-1], agree[-1]), fontsize=7, xytext=(4, -9), textcoords="offset points")
    ax.set_xlabel("equal-opportunity gap vs proxy reference (lower is better)")
    ax.set_ylabel("agreement with proxy reference (%)")
    ax.set_title("Pareto front: correction strength lambda from 0 to 1")
    ax.legend(); fig.tight_layout(); fig.savefig(ROOT / "pareto_front.png", dpi=130)
    for lam in (0, .5, 1):
        a, g = metrics(top(lam * neutral + (1 - lam) * raw))
        print(f"lambda={lam:.1f}  agreement {a:.2f}%  EO gap {g:.3f}")
    print("pareto_front.png written")


if __name__ == "__main__":
    main()
