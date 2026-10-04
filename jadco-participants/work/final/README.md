# Collection Équinoxe — estimer la hausse de loyer 2026 (JADCO « Clés en main », CodeML 2026)

<!-- GENERATED RESULTS START -->
Livrable principal : **`equinoxe_hausse_2026.ipynb`** (174 cellules exécutées).

## Résultat calculé (`outputs/final_answer.json` fait foi)

| Mesure | Résultat |
|---|---|
| Définition | Croissance à unité constante du loyer effectif (net des concessions), médiane des paires, baux débutant en 2026 (renouvellements + relocations) |
| Hausse effective | **6,01 %**, bande prédictive : 4,74 à 7,26 % |
| Hausse contractuelle | **4,72 %**, bande : 3,66 à 6,22 % |
| Moyenne effective pondérée par le loyer | 6,72 % |
| Ancrage de la politique commune (calendrier TAL ; proxy pour The Met) | 3,78 % |
| Erreur absolue moyenne du backtest récent (effectif / contractuel) | 0,30 / 0,58 point |
| Biais moyen récent (prévu − réalisé, effectif) | 0,30 point |
| Sensibilité si ce biais était soustrait (non validée) | 5,71 % |
| Sensibilité aux pondérations de politique | 5,85 à 6,63 % |
| Sensibilité aux réglages déclarés | 5,92 à 6,52 % |

Le headline est la médiane du modèle structurel. La bande ajoute son erreur quadratique historique,
biais inclus ; elle ne constitue pas une garantie de couverture. Les règles ont été repérées sur les
années également utilisées pour le backtest, donc cette validation est favorable au modèle.
Le précédent de concessions du début de l'historique est affiché. Les poids uniformes restent un
choix de jugement ; The Met utilise une hypothèse de politique commune, pas le droit québécois.
<!-- GENERATED RESULTS END -->

## Principes de construction (discipline d'un desk quantitatif)

* **Aucun nombre magique** : chaque paramètre est estimé sur les données (à la date de coupure), tiré d'une source publique datée, ou déclaré comme a priori dans une cellule `# CONFIG` avec un test de sensibilité.
* **Aucun chiffre tapé dans le texte** : les paragraphes « Lecture » sont rédigés par le code à partir des valeurs calculées (`narrate`), donc le texte ne peut pas diverger des données.
* **Information datée (point-in-time)** : chaque valeur publique porte `published_on` ; le backtest rejoue trois millésimes (décembre, janvier, en direct) sans fuite.
* **Même code pour la prévision et le backtest** : `backtest()` appelle exactement la fonction qui produit `estimate_2026()`.
* **Incertitude décomposée** : échantillonnage, structure (grille des inconnues de 2026) et erreur de modèle (backtest), plus le risque de régime rapporté à part.

## Commande de livraison

Depuis ce dossier, `python release.py --rebuild` exécute le notebook, contrôle la qualité, lance les tests, actualise les documents et les graphiques du jury, puis écrit `../submission/jadco_submission.zip`.
Le ZIP contient uniquement les fichiers du livrable, la présentation et les tables publiques. Son `MANIFEST.json` permet de vérifier chaque empreinte SHA-256.
Les sorties sont produites dans un dossier temporaire puis publiées après exécution réussie ; une cellule en erreur conserve le notebook et les résultats précédents.
Pour vérifier et empaqueter le notebook déjà exécuté : `python release.py`.

## Comment exécuter

1. **Python 3.13** et les versions de `requirements.txt` :
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   python -m ipykernel install --prefix "$VIRTUAL_ENV" --name python3
   ```
2. **Données CRM (non livrées, jamais à publier)** : placez les quatre CSV fournis (`equinoxe_listings.csv`, `equinoxe_lease_history.csv`, `equinoxe_concessions.csv`, `equinoxe_asking_history.csv`) dans un dossier, puis pointez vers lui :
   ```bash
   export JADCO_DATA_DIR=/chemin/vers/les/csv     # par défaut : ../..
   ```
3. **Exécuter** (moins d'une minute) :
   ```bash
   jupyter nbconvert --to notebook --execute equinoxe_hausse_2026.ipynb --output equinoxe_hausse_2026.ipynb
   ```
   ou, depuis les sources : `python build_notebook.py --execute`. Après un build réussi : `python refresh_docs.py`, puis `python ../presentation/make_presentation.py`.
4. **Vérifier** la qualité et la confidentialité :
   ```bash
   python lint_notebook.py equinoxe_hausse_2026.ipynb      # aucun littéral hors CONFIG, aucun chiffre tapé dans le texte, dates par une seule fonction,
                                                            # Markdown avant chaque étape, mauvaise clé, nommage, noms morts, sorties CRM
   python -m pytest tests -q                                # exécution, contrôles de validité, aucune donnée CRM, cohérence de la réponse
   ```
5. *(facultatif)* Régénérer les tables publiques : `python external/fetch.py` (télécharge les CSV de Statistique Canada).

Les fichiers de `external/tidy/` (données **publiques**) sont livrés, donc le notebook s'exécute sans accès réseau.

## Contenu du dossier

| Fichier | Rôle |
|---|---|
| `equinoxe_hausse_2026.ipynb` | notebook évalué, exécuté |
| `nb_part01.py … nb_part12.py`, `build_notebook.py` | sources du notebook (format « percent ») et son assemblage |
| `release.py`, `outputs/validation.json` | chaîne de livraison, versions réellement utilisées et empreintes des résultats |
| `lint_notebook.py`, `tests/` | grille de qualité automatisée (stricte) et tests |
| `external/` | `fetch.py`, `manual_public_rates.csv` (valeurs saisies à la main avec URL, date de publication et confiance), `SOURCES.md` (licences), `tidy/` (tables publiques avec `published_on`) |
| `outputs/model_2026.json` | **modèle entraîné** (XGBoost, défi indépendant ; non retenu comme prévision, section 10h) |
| `outputs/aggregates_2026.csv`, `outputs/dashboard_2026.html` | prévision par propriété × chambres, **agrégats seulement**, tableau de bord |
| `outputs/evidence.json` | tables agrégées qui alimentent les graphiques du jury et les contrôles |
| `outputs/final_answer.json` | la réponse finale, sa définition, sa décomposition, ses scénarios et l'erreur du backtest |

## Où trouver chaque critère du barème

| Critère | Section du notebook |
|---|---|
| Définition de la hausse (10) | 1 (choix), 7 (tableau comparatif des cinq définitions et critères) |
| Analyse des données et effet de mix (15) | 2 (audit « description contre réalité », renommage), 3 (panel fixe, décomposition shift-share exacte par propriété, indice de Fisher) |
| Appariement à unité constante (15) | 4 (clé `prop_code + unit_code`, entonnoir de validité, **démonstration de l'échec de `site + unit_code`**, sensibilité aux filtres) ; 9 (indice de loyers répétés et effets fixes d'unité) |
| Concessions (10) | 5 (rattachement des lignes, **reconstruction de `rent_effective`**, PromoPay = forfait unique, décomposition pénétration × profondeur de l'écart contractuel − effectif) |
| Renouvellements vs relocations (10) | 6 (2 × 2 province × type, règles du TAL et de l'Ontario, art. 1896/1950/1955 C.c.Q., **une seule politique de la société** (6c bis), **The Met exempté** : trois tests) |
| Données externes (15) | 8 (IPC, SCHL, TAL, Ontario ; `published_on` ; réconciliation avec erreurs « un an de côté » ; niveaux ; concessions contre inoccupation) |
| Prévision et backtest (15) | 10 (modèle structurel, cohorte réelle simulée, indexation sur l'IPC (10c), **réforme du TAL datée par l'avis** (10d), `estimate_2026`/`backtest` (10e), trois millésimes sans fuite et références naïves (10f), XGBoost comparé (10h)) |
| Qualité du notebook (10) | tout le notebook : Markdown avant chaque étape, aucun nombre magique ni chiffre tapé dans le texte (`lint_notebook.py` : 0 problème), contrôles de validité exécutés |
| Bonus | Ontario (6d, 11d : valeur de l'exemption), prévision par propriété × chambres (11e), intervalles, scénarios et sensibilité aux a priori (11b-11c), tableau de bord (11e), aperçu 2027 (11g) |

## Hypothèses, limites, références
Voir les sections 12 et 13 du notebook (hypothèses numérotées, limites chiffrées, sources publiques avec URL, méthodes, **outils d'IA utilisés**).

## Outils d'intelligence artificielle (déclaration)
Claude Code (Anthropic, modèle Claude Sonnet 5.5) a servi à écrire et tester le code, à explorer les données, à rechercher les sources publiques et à rédiger. Une seconde session Claude Code (modèle Claude Opus 5.5) a mené des analyses indépendantes de contre-vérification (simulation de cohorte, XGBoost, loyers répétés, modèle hédonique), des relectures et la refonte finale (paramètres estimés ou déclarés, texte chiffré rédigé par le code, ancrage du TAL daté par l'avis, concessions à deux marges). Tout chiffre du notebook est produit par le code exécuté ; les valeurs saisies à la main sont dans `external/manual_public_rates.csv` avec leur source et leur niveau de confiance. Codex (OpenAI) a repris le travail interrompu pour reconstruire, vérifier et corriger la cohérence des livrables. L'équipe valide les choix de méthode.

## Confidentialité
Le sous-ensemble CRM n'est ni livré, ni publié, ni versionné ; le notebook n'écrit que des agrégats (`outputs/`), vérifiés par `tests/`.
