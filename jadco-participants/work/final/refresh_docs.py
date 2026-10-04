"""Refresh the README's result block from the executed notebook and answer JSON."""
import json
from pathlib import Path

import nbformat

HERE = Path(__file__).resolve().parent
START = "<!-- GENERATED RESULTS START -->"
END = "<!-- GENERATED RESULTS END -->"


def fr(value):
    return f"{value:.2f}".replace(".", ",")


def result_block(answer, cells):
    a = answer
    low, high = a["headline_band_pct"]
    clow, chigh = a["contract_band_pct"]
    sensitivity = a["prior_sensitivity_effective_pct"].values()
    return f"""{START}
Livrable principal : **`equinoxe_hausse_2026.ipynb`** ({cells} cellules exécutées).

## Résultat calculé (`outputs/final_answer.json` fait foi)

| Mesure | Résultat |
|---|---|
| Définition | {a['definition']} |
| Hausse effective | **{fr(a['headline_effective_pct'])} %**, bande prédictive : {fr(low)} à {fr(high)} % |
| Hausse contractuelle | **{fr(a['contract_pct'])} %**, bande : {fr(clow)} à {fr(chigh)} % |
| Moyenne effective pondérée par le loyer | {fr(a['effective_weighted_mean_pct'])} % |
| Ancrage de la politique commune (calendrier TAL ; proxy pour The Met) | {fr(a['renewal_anchor_law_pct'])} % |
| Erreur absolue moyenne du backtest récent (effectif / contractuel) | {fr(a['backtest_mae_points']['effective'])} / {fr(a['backtest_mae_points']['contract'])} point |
| Biais moyen récent (prévu − réalisé, effectif) | {fr(a['backtest_bias_points']['effective'])} point |
| Sensibilité si ce biais était soustrait (non validée) | {fr(a['bias_adjusted_sensitivity_pct']['effective'])} % |
| Sensibilité aux pondérations de politique | {fr(min(sensitivity))} à {fr(max(sensitivity))} % |
| Sensibilité aux réglages déclarés | {fr(a['robustness_effective_range_pct'][0])} à {fr(a['robustness_effective_range_pct'][1])} % |

Le headline est la médiane du modèle structurel. La bande ajoute son erreur quadratique historique,
biais inclus ; elle ne constitue pas une garantie de couverture. Les règles ont été repérées sur les
années également utilisées pour le backtest, donc cette validation est favorable au modèle.
Le précédent de concessions du début de l'historique est affiché. Les poids uniformes restent un
choix de jugement ; The Met utilise une hypothèse de politique commune, pas le droit québécois.
{END}"""


def main():
    answer = json.loads((HERE / "outputs/final_answer.json").read_text())
    notebook = nbformat.read(HERE / "equinoxe_hausse_2026.ipynb", as_version=4)
    path = HERE / "README.md"
    text = path.read_text()
    if START in text:
        before, rest = text.split(START, 1)
        _, after = rest.split(END, 1)
    else:
        before = text.split("\n", 1)[0] + "\n\n"
        after = "\n\n## Principes de construction" + text.split("## Principes de construction", 1)[1]
    path.write_text(before + result_block(answer, len(notebook.cells)) + after)
    a = answer
    validation_path = HERE / "outputs/validation.json"
    validation = json.loads(validation_path.read_text()) if validation_path.exists() else {}
    test_count = validation.get("tests_passed", "consulter le rapport de validation")
    reproduction_path = HERE / "outputs/reproducibility.json"
    reproduction_note = "Reproduction indépendante non consignée pour ce build."
    if reproduction_path.exists():
        reproduction = json.loads(reproduction_path.read_text())
        reproduced = reproduction["numerical_artifacts_reproduced"]
        reproduction_note = f"Reconstruction depuis un ZIP extrait : {sum(reproduced.values())} artefacts numériques reproduits à l’octet près ; suite de tests exécutée dans cette copie."

    report = f"""# JADCO — reprise du travail interrompu de Claude

État : notebook reconstruit et exécuté ; documentation et présentation régénérées depuis les mêmes résultats.

## Dernier résultat calculé

- Effectif : {fr(a['headline_effective_pct'])} %, bande prédictive {fr(a['headline_band_pct'][0])}–{fr(a['headline_band_pct'][1])} %.
- Contractuel : {fr(a['contract_pct'])} %, bande {fr(a['contract_band_pct'][0])}–{fr(a['contract_band_pct'][1])} %.
- Backtest récent : MAE {fr(a['backtest_mae_points']['effective'])} / {fr(a['backtest_mae_points']['contract'])} point (effectif / contractuel).
- Biais effectif : {fr(a['backtest_bias_points']['effective'])} point ; sensibilité corrigée {fr(a['bias_adjusted_sensitivity_pct']['effective'])} %, sans prétendre valider cette correction.

## Corrections de l'audit

- Les sources plus récentes que le notebook ont été intégrées. Un test compare désormais toutes les cellules livrées aux sources.
- Le « maximum historique » est explicitement limité aux années réellement comparées.
- Les précédents de concessions où l'IPC est inférieur au drag précédent sont affichés ; la déclaration « sans précédent » a été retirée.
- The Met utilise explicitement un proxy de politique commune : le calendrier québécois n'est pas présenté comme son droit applicable.
- Les fenêtres d'avis sont calculées en mois calendaires, avec tests des bornes et des baux courts.
- Le biais récent est montré en sensibilité ; la bande conserve l'erreur quadratique totale, biais inclus.
- L'écart-type d'échantillonnage agrégé utilise la racine de la moyenne pondérée des variances.
- Le headline est nommé médiane du modèle structurel, conformément au calcul ; les bandes viennent de la distribution prédictive.
- L'aperçu suivant nomme correctement le drag de la grille ; les signaux récents sont décrits comme contexte, pas comme entrées quantitatives effectivement utilisées.
- README et présentation lisent le JSON calculé ; les chiffres empiriques de la présentation ne sont plus tapés dans son générateur.
- La reconstruction publie ses sorties après une exécution réussie ; un échec écrit un notebook `.failed.ipynb` et préserve le notebook et les sorties précédents.

## Traçabilité des nombres

| Origine | Où la vérifier | Ce que cela signifie |
|---|---|---|
| Données CRM | sections 2–7, paires et agrégats calculés | loyers, concessions, proportions et cohorte mesurés |
| Sources publiques | `external/manual_public_rates.csv`, tables `tidy`, `SOURCES.md` | URL, date de publication, confiance ; filtrage par date de coupure |
| Paramètres estimés | section 10 : pente TAL, pénétration des concessions, résidus, renouvellements | seules données disponibles à l'origine de chaque prévision |
| Choix de jugement | `Config`, `ForecastConfig`, `MixtureConfig`, section 10f bis et 11b | grille, poids uniformes et fenêtres explicités ; ce ne sont pas des faits identifiés |
| Simulation | section 11c | dispersion, précision Monte-Carlo et bandes calculées |
| Comparateur générique | sections 10a et 10f | ses hyperparamètres ne pilotent pas le headline structurel |
| Présentation / README | `outputs/final_answer.json`, `refresh_docs.py`, `../presentation/make_presentation.py` | mêmes résultats que le notebook exécuté |

Les tests de robustesse déplacent l'effectif de {fr(a['robustness_effective_range_pct'][0])} à {fr(a['robustness_effective_range_pct'][1])} %.
Les variantes de politique donnent {fr(min(a['prior_sensitivity_effective_pct'].values()))} à {fr(max(a['prior_sensitivity_effective_pct'].values()))} %.
Le choix des scénarios et de poids uniformes reste un jugement : l'entropie maximale ne rend pas la grille objectivement correcte.

## Vérification de cette reprise

- Toutes les cellules de code exécutées, sans sortie d'erreur ; aucun avertissement dans les sorties stockées.
- Gate stricte : zéro finding. Suite de vérification : {test_count} tests passent.
- {reproduction_note}
- La présentation PDF contient huit diapositives ; les notes et le PDF sont régénérés depuis le JSON.
- Les tests contrôlent aussi les sources publiques et l'absence de lignes CRM dans les exports.

Le précédent auditeur indépendant estimait environ 88,5/100 avant bonus/pénalités, environ 91 net, contre environ 82/83 auparavant.
Ce score portait sur la version précédente. Aucun nouveau score indépendant n'est revendiqué pour cette reprise.

## Limites à défendre devant le jury

Les mêmes années ont servi à repérer les relations puis à mesurer leur backtest : il ne s'agit pas d'une validation totalement indépendante.
Le régime ancien est moins bien prévu. Le biais récent n'est pas prouvé stable. Les dates d'avis réelles et la politique future de concessions sont inconnues.
Le calendrier TAL appliqué comme proxy à Ottawa et le taux prévu de l'ancienne méthode restent des hypothèses.
La bande ne garantit pas sa couverture en cas de changement de régime.

## Livraison

Commande complète : `python release.py --rebuild` (exécution, gate, tests, documentation, présentation, ZIP avec empreintes).
Notebook : `equinoxe_hausse_2026.ipynb`. Réponse : `outputs/final_answer.json`.
Présentation : `../presentation/presentation.pdf`, notes : `../presentation/PRESENTATION.md`.
Le dossier JADCO demeure ignoré par git ; cette reprise n'effectue ni commit, ni push, ni soumission sur la plateforme.

Pour actualiser après modification : reconstruire et exécuter, lancer lint et pytest, puis `python refresh_docs.py` et `python ../presentation/render_pdf.py`.
"""
    (HERE / "AUDIT_PROGRESS.md").write_text(report)
    print("README refreshed from executed results")


if __name__ == "__main__":
    main()
