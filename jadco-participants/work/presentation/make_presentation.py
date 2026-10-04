"""Génère le plan de présentation devant le jury à partir des résultats calculés (outputs/final_answer.json).

Aucun chiffre n'est tapé ici : tout vient du fichier produit par le notebook. Relancer après chaque reconstruction du notebook.
Usage : python make_presentation.py [chemin/vers/final_answer.json]      → écrit PRESENTATION.md à côté du script.
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ANSWER_PATH = HERE.parent / "final" / "outputs" / "final_answer.json"


def fr(value, digits=1):
    """Nombre au format français (virgule décimale)."""
    return f"{value:.{digits}f}".replace(".", ",")


def build(answer):
    low, high = answer["headline_band_pct"]
    contract_low, contract_high = answer["contract_band_pct"]
    scenarios = answer["scenarios_effective_pct"]
    diagnostics = answer["presentation_diagnostics"]
    mae = answer["backtest_mae_points"]
    scenario_lines = "\n".join(f"  - {name} : {fr(value)} %" for name, value in scenarios.items())
    return f"""# Collection Équinoxe — hausse de loyer 2026 : plan de présentation (8 minutes, 8 diapositives)

*Fil conducteur du défi : Définir → Mesurer → Tester → Défendre. Les chiffres viennent de `outputs/final_answer.json` ; ne pas les retaper.*

## 1. Composition contre hausse des prix (45 s)
- La médiane des loyers de tous les baux monte de {fr(diagnostics["naive_growth_pct"])} % en {diagnostics["mix_year"]} : c'est la **composition** (Le Carlyle et The Met entrent).
- Même panel d'immeubles : {fr(diagnostics["fixed_panel_growth_pct"])} %. Indice de Fisher : {fr(diagnostics["fisher_growth_pct"])} %. Décomposition exacte : l'entrée de nouveaux immeubles explique l'essentiel.
- *Visuel : section 3 du notebook (panel fixe contre tous les baux, puis la cascade).*

## 2. Notre définition, en une phrase (45 s)
- **{answer["definition"]}**
- Toujours publiée à côté : la hausse **contractuelle** (seule comparable au TAL). Pourquoi deux nombres : les concessions ({fr(diagnostics["concession_penetration_pct"])} % des baux de la dernière année observée) séparent le loyer au bail du revenu encaissé.
- *Visuel : section 7, le tableau comparatif des cinq définitions.*

## 3. Même unité, bonne clé (45 s)
- Clé `prop_code + unit_code`. Démonstration de l'échec de `site + unit_code` : fausses paires entre appartements différents, dispersion accrue, vraies paires détruites.
- Trois estimateurs indépendants (médiane des paires, loyers répétés, effets fixes d'unité) servent de contre-vérification ; leurs écarts sont chiffrés dans le notebook.

## 4. Contractuel contre effectif : l'histoire des concessions (1 min)
- `PromoPay` est un forfait unique, pas un frais mensuel : `rent_effective` se reconstruit à ≤ {fr(diagnostics["reconstruction_tolerance_dollars"])} $ pour {fr(diagnostics["reconstruction_share_pct"])} % des baux.
- Dans le régime récent, le drag annuel moyen suit **l'IPC loyers du Québec de l'année précédente**, sans paramètre. C'est ce qui explique l'écart de 2024-2025.

## 5. Renouvellements, relocations, Québec, Ontario (1 min)
- Les renouvellements suivent le TAL depuis 2023 seulement ; avant, non. Les relocations suivent l'IPC loyers de l'année précédente.
- The Met (Ottawa) est **exempté** de la ligne directrice (immeuble livré en 2023) : trois tests. Ses renouvellements ressemblent à ceux du Québec : politique de la société, pas un plafond. Valeur de l'exemption chiffrée.

## 6. Ce que dit le marché (45 s)
- IPC, SCHL, TAL, Ontario : chaque série porte sa date de publication (`published_on`), donc le backtest n'utilise que ce qui était public.
- Les relocations et concessions suivent récemment l'IPC loyers de l'année précédente. Le TAL est un ancrage de politique récent, avec un cadre juridique propre au Québec. Les niveaux sont comparés séparément à la SCHL, sans confondre cette comparaison avec la croissance.

## 7. La prévision et sa validation (1 min 30)
- Modèle par composantes + simulation de la cohorte réelle de {answer["cohort_units"]} unités ; backtest 2023-2024-2025 avec données tronquées et sources datées, trois millésimes, références naïves toujours affichées.
- Erreur absolue moyenne récente : {fr(mae["effective"])} point (effectif), {fr(mae["contract"])} point (contractuel).
- Honnêteté : bon dans le régime 2023-2025, **pas universel** (2021-2022) ; les règles ont été repérées sur ces années ; XGBoost testé et **rejeté**.

## 8. Réponse 2026 (1 min)
- **Hausse effective : {fr(answer["headline_effective_pct"])} %** [bande 10-90 % : {fr(low)} à {fr(high)} %], {answer["headline_method"]}.
- **Hausse contractuelle : {fr(answer["contract_pct"])} %** [{fr(contract_low)} à {fr(contract_high)} %]. Ancrage du TAL daté par l'avis : {fr(answer["renewal_anchor_law_pct"], 2)} %.
- Scénarios (effectif) :
{scenario_lines}
- Les coins de politique sont des scénarios, pas des observations. Le calendrier TAL est un proxy de politique pour The Met ; ce n’est pas le droit applicable à Ottawa.
- Sensibilité au biais récent : {fr(answer["bias_adjusted_sensitivity_pct"]["effective"])} % si le biais historique était soustrait ; cette correction n’est pas validée.
- **Message final : 2026 se joue dans la politique de concessions.** Chaque point de drag en plus retire environ un point à la hausse effective (chiffré dans la section 10g du notebook).

---
### Questions probables du jury
- *Pourquoi l'effectif et pas le contractuel ?* Le revenu encaissé ; les deux sont publiés.
- *Votre backtest est-il indépendant ?* Les règles ont été repérées sur l'historique qui sert aussi à les tester ; nous n'en faisons pas une preuve de performance future.
- *Pourquoi des poids uniformes ?* Jugement explicite faute de preuve suffisante ; la sensibilité aux autres lectures est affichée.
- *Pourquoi conserver le biais récent ?* Il est montré et inclus dans l'erreur quadratique ; le soustraire à partir des mêmes quelques années ne serait pas une correction indépendante.
- *Et si l'indexation des concessions s'arrête ?* Scénario cliquet ; le drag n'a jamais baissé dans les données, mais le moteur IPC est déjà passé sous le drag précédent au début de l’historique.
- *Données CRM ?* Aucune dans les livrables : seuls des agrégats sont écrits (tests automatisés).
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("answer", nargs="?", type=Path, default=ANSWER_PATH)
    arguments = parser.parse_args()
    answer = json.loads(arguments.answer.read_text(encoding="utf-8"))
    required = {"definition", "headline_effective_pct", "headline_band_pct", "headline_method", "contract_pct", "contract_band_pct", "renewal_anchor_law_pct",
                "presentation_diagnostics", "backtest_mae_points", "bias_adjusted_sensitivity_pct", "scenarios_effective_pct", "cohort_units"}
    missing = required - set(answer)
    if missing:
        sys.exit(f"final_answer.json incomplet (clés absentes : {sorted(missing)}) : reconstruire le notebook d'abord.")
    (HERE / "PRESENTATION.md").write_text(build(answer), encoding="utf-8")
    print("PRESENTATION.md écrit depuis", arguments.answer)
