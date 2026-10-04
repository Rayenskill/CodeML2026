# JADCO — reprise du travail interrompu de Claude

État : notebook reconstruit et exécuté ; documentation et présentation régénérées depuis les mêmes résultats.

## Dernier résultat calculé

- Effectif : 6,01 %, bande prédictive 4,74–7,26 %.
- Contractuel : 4,72 %, bande 3,66–6,22 %.
- Backtest récent : MAE 0,30 / 0,58 point (effectif / contractuel).
- Biais effectif : 0,30 point ; sensibilité corrigée 5,71 %, sans prétendre valider cette correction.

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

Les tests de robustesse déplacent l'effectif de 5,92 à 6,52 %.
Les variantes de politique donnent 5,85 à 6,63 %.
Le choix des scénarios et de poids uniformes reste un jugement : l'entropie maximale ne rend pas la grille objectivement correcte.

## Vérification de cette reprise

- Toutes les cellules de code exécutées, sans sortie d'erreur ; aucun avertissement dans les sorties stockées.
- Gate stricte : zéro finding. Suite de vérification : 36 tests passent.
- Reconstruction depuis un ZIP extrait : 4 artefacts numériques reproduits à l’octet près ; suite de tests exécutée dans cette copie.
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
