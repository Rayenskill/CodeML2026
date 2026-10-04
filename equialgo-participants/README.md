# ÉquiAlgo (IVADO): fair student financing

A scholarship-scoring model grants awards to 48.4% of applicants from Montréal and the
Capitale-Nationale against 27.3% from three remote regions. The task: diagnose the bias, correct it,
and propose a monitoring plan. All data is synthetic. Full rules: `consignes-fr.pdf`, `consignes-en.pdf`.

## Constraints

- **Budget.** The grant rate on the 4,000 evaluation applicants must be between 36% and 44%, otherwise
  the technical section scores zero.
- **`decision_octroi` is not the target.** Scoring uses a hidden reference standard built independently
  of the historical committee.
- **Dropping `region_administrative` is not enough.** Distance, hours worked, income and postal code
  carry regional information.

| Section | Points | Judged by |
|---|---|---|
| Diagnostic rigour | 25 | jury |
| Technical solution (equity 20, utility 15) | 35 | automated scorer |
| Governance and ethics | 25 | jury |
| Pitch and code quality | 15 | jury |

## Our solution

A sparse-spline logistic model of the historical committee, scored counterfactually: each applicant
keeps their own cote R and working hours, every other feature is replaced by common reference profiles.
The score is averaged over 30 seeds and the top 40% is granted (1,600 of 4,000; remote and centre both
at 40.0%). Platform preview: 94.83% accuracy, 94.61% macro F1. Details, evidence and limits:
[`MODEL_LOGIC.md`](MODEL_LOGIC.md).

## Run

```bash
python -m venv venv
venv\Scripts\activate             # Linux / macOS: source venv/bin/activate
pip install -r requirements.txt
python model_corrige.py           # writes ../predictions.csv (repository root) and pareto_front.png
```

## Files

| Path | Contents |
|---|---|
| `../predictions.csv` | At the repository root, as the rules require. Submission: `id_candidat,decision_octroi`, 4,000 rows |
| `model_corrige.py` | Mitigation and Pareto front; uses `work/clean95/seeded_model.py` |
| `audit_rapport.ipynb` | Audit, proxy variables, model selection, graphs |
| `MODEL_LOGIC.md` | What the model does and why, validation, monitoring plan |
| `model_ensemble.py` | Entry point of the model-selection experiments in `work/codex_model/` |
| `baseline_model.ipynb` | Organizers' production model and fairness audit |
| `data/` | Not in the repository. Put the organizers' `donnees_demandes.csv` (10,000 labelled) and `candidats_evaluation.csv` (4,000 to score) here before running |
| `work/clean95/` | Final seed-ensembled model, validation report, notebook builder |
| `work/codex_model/` | Candidate models, reports, preview results (`platform_results.csv`) |
| `work/agent_dgp/` | Analysis of how the synthetic data was generated |

## Présentation PowerPoint : quoi mettre

Pitch de 5 minutes devant le jury (`presentation.pdf` est le support demandé par les consignes). Le jury note
65 points sur 100 : rigueur du diagnostic (25), gouvernance et éthique (25), pitch et qualité du code (15). Les 35
autres points viennent du correcteur automatique, mais il faut quand même les expliquer. Viser 8 diapositives,
environ 35 secondes chacune. Tous les chiffres ci-dessous viennent de `MODEL_LOGIC.md` et de `audit_rapport.ipynb` :
ne pas en citer d'autres.

### Diapositive 1 — Titre (15 s)
- Nom du défi (ÉquiAlgo, IVADO), nom de l'équipe, membres.
- Une phrase : « Un modèle de bourses qui juge le mérite, pas le code postal. »

### Diapositive 2 — Le problème (30 s)
- Le comité accorde une bourse à 48,4 % des candidats des grands centres contre 27,3 % dans les régions éloignées
  (écart de 0,211).
- La cote R moyenne (27,3 contre 28,0) n'explique qu'une partie de cet écart.
- L'enveloppe est fixe : entre 36 % et 44 % d'octrois, sinon la section technique vaut zéro.
- `decision_octroi` n'est pas la cible : on est noté contre un étalon caché, pas contre le comité.
- Visuel : deux barres (centres, régions éloignées).

### Diapositive 3 — Diagnostic : d'où vient le biais (50 s) → rigueur du diagnostic
- Variables proxys : la distance identifie la région presque parfaitement (AUC 0,998) ; les heures travaillées
  (0,81) et le revenu (0,69) portent aussi un signal régional.
- Supprimer `region_administrative` ne change rien (écart 0,220 → 0,220). Supprimer aussi le code postal : 0,183 ;
  aussi la distance : 0,118 ; modèle sur cote R et heures seulement : 0,070.
- Conclusion à dire : supprimer des colonnes soit ne fait rien, soit jette du signal légitime.
- Le comité a quatre moteurs réels : cote R (dominant), heures, log du revenu, et l'indicateur « région éloignée »
  avec un poids nettement négatif (une pénalité régionale). Programme et première génération : indiscernables de
  zéro.
- Visuel : le graphique des écarts par jeu de colonnes (section 1 à 3 du notebook).

### Diapositive 4 — Métrique d'équité choisie et pourquoi (35 s) → rigueur du diagnostic
- Parité démographique et égalité des chances ne peuvent pas tenir ensemble quand les profils diffèrent : il faut
  en choisir une et le défendre.
- Notre choix de politique, dit explicitement : mérite = performance académique (cote R) + effort (heures
  travaillées). Le revenu et la région sont traités comme des nuisances. C'est un choix de valeur, pas un résultat
  statistique.

### Diapositive 5 — Notre solution (50 s) → solution technique
- Étape 1 : régression logistique à splines parcimonieuse qui modélise le comité (10 000 demandes historiques).
- Étape 2 : score contrefactuel. Chaque candidat garde sa cote R et ses heures ; toutes les autres variables sont
  remplacées par 256 profils de référence communs, et on moyenne les probabilités.
- Étape 3 : ensemble de 30 graines (rééchantillonnage bootstrap et profils propres à chaque graine).
- Étape 4 : on accorde aux 40 % les mieux classés (1 600 sur 4 000). Égalités départagées par la cote R puis les
  heures, jamais par un identifiant.
- Pourquoi ce modèle : sur plus de 33 configurations, l'écart d'exactitude est d'environ 0,8 point et les six
  meilleures tiennent en 0,0003 de log loss (0,2566 à 0,2569) ; les arbres n'apportent rien.
- Jamais utilisés : identifiant, ordre des lignes, scores de la plateforme, étiquettes reconstruites.
- Visuel : schéma en quatre boîtes.

### Diapositive 6 — Résultats et front de Pareto (45 s) → solution technique
- Taux d'octroi régions éloignées / centres : 40,0 % / 40,0 % (sans correction : 27,8 % / 48,4 %).
- Aperçu de la plateforme : 94,83 % d'exactitude, 94,61 % de F1 macro.
- Insérer `pareto_front.png` : on balaie la force de correction (lambda) et le budget. À lambda 0, accord 89,40 %
  et écart d'égalité des chances 0,300 ; à 0,5 : 93,55 % et 0,179 ; à 1,0 : 99,35 % et 0,002.
- Dire que ce front est mesuré contre une référence de substitution (mérite = z(cote R) + 0,2·z(heures)), pas contre
  l'étalon du jury : il montre la forme du compromis.

### Diapositive 7 — Stabilité et limites (40 s) → rigueur, honnêteté
- 97,0 % des candidats reçoivent la même décision des 30 graines (120 votes partagés sur 4 000).
- Deux ensembles indépendants de 30 graines diffèrent sur 8 décisions.
- Exactitude hors échantillon sur le comité : 89,3 % ± 0,6 ; AUC 0,959 ± 0,004.
- Limites à dire soi-même : l'étalon est inconnu ; un écart-type sur une exactitude proche de 95 % vaut environ
  0,35 point, donc nos variantes (94,33 % à 94,88 %) sont presque indiscernables ; environ 5 % des étiquettes de
  référence ressemblent à du bruit irréductible ; la définition du mérite est un choix.
- Transparence : les scores d'aperçu ont servi à choisir entre familles de modèles, jamais à ajuster une décision
  individuelle.

### Diapositive 8 — Gouvernance et surveillance en production (50 s) → gouvernance et éthique
- Chaque semaine : taux d'octroi global et par groupe, alerte si l'écart régions / centres dépasse 3 points.
- Chaque mois : dérive de la cote R, des heures, du revenu (PSI) et de la distribution du score.
- Chaque trimestre : réentraînement avec de nouvelles graines, au moins 99 % d'accord exigé entre deux jeux de
  graines.
- Par décision : journaliser le score, la cote R et les heures utilisées, pour pouvoir expliquer et contester.
- Revue humaine de la bande limite (environ 3 % des candidats, ceux dont les graines votent différemment).
- Qui décide : le modèle classe, l'institution fixe le budget et la définition du mérite.

### Questions probables du jury
- Pourquoi ne pas simplement retirer la région ? Voir diapositive 3 : 0,220 → 0,220.
- Pourquoi les heures travaillées comptent-elles comme du mérite ? Choix de politique : l'effort compense la cote R
  perdue au travail rémunéré.
- Votre 94,83 % est-il surajusté à la plateforme ? Sélection légère sur un nombre bruité, déclarée ; le pipeline
  ne s'entraîne que sur l'historique.
- Que se passe-t-il si l'étalon pondère le revenu ? Les résultats changent ; c'est une limite déclarée.

### À ne pas faire
- Ne pas présenter d'anciens scores obtenus en décodant les étiquettes cachées via la plateforme : ce n'est pas le
  modèle soumis.
- Ne pas dire « exactitude de 95 % » : le chiffre mesuré est 94,83 %.
- Ne pas présenter le front de Pareto comme la note du jury.
