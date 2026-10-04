# JADCO — Collection Équinoxe

Le point d’entrée de livraison est [`work/final/README.md`](work/final/README.md).
Le dossier `work/final/` contient le notebook exécuté, ses sources, les tables publiques, le modèle comparateur entraîné et les contrôles.
Les CSV CRM restent à la racine : ils ne sont jamais inclus dans le bundle.

## Prêt pour le jury

- [`work/submission/jadco_submission.zip`](work/submission/jadco_submission.zip) : bundle vérifié, avec manifeste SHA-256.
- [`presentation.pdf`](presentation.pdf) : huit diapositives avec graphiques calculés (copie de `work/presentation/presentation.pdf`, régénérée par `release.py`).
- [`work/presentation/PRESENTATION.md`](work/presentation/PRESENTATION.md) : notes de présentation et réponses aux questions du jury.
- [`work/final/outputs/final_answer.json`](work/final/outputs/final_answer.json) : estimation, définition, bandes et scénarios.
- [`work/final/AUDIT_PROGRESS.md`](work/final/AUDIT_PROGRESS.md) : provenance, corrections et limites.

## Reconstruire et vérifier

Avec l’environnement du projet activé :

```bash
cd work/final
python release.py --rebuild
```

Cette commande exécute le modèle, lance le gate et les tests, régénère les documents et le PDF, puis reconstruit le ZIP.
Pour valider et empaqueter le notebook déjà exécuté : `python release.py`.
`outputs/validation.json` consigne les versions réellement utilisées, les contrôles et les empreintes des résultats.

## Organisation

`work/external/` conserve les sources publiques et leurs scripts de collecte ; `work/final/external/` livre les tables publiques nécessaires hors réseau.
`archive/` conserve les stratégies et prototypes exploratoires ; ils ne sont pas des entrées supportées.
Les données CRM, le notebook de départ et les consignes d’origine restent à leur emplacement.

## Présentation PowerPoint : quoi mettre

Présentation de 8 minutes, 8 diapositives, fil conducteur : Définir → Mesurer → Tester → Défendre. Le support
existe déjà (`presentation.pdf`) et le plan détaillé avec les minutages est dans
[`work/presentation/PRESENTATION.md`](work/presentation/PRESENTATION.md). Tous les chiffres viennent de
`work/final/outputs/final_answer.json` : ne pas les retaper à la main, relancer `make_presentation.py`. Barème du
notebook : définition 10, analyse et effet de mix 15, appariement à unité constante 15, concessions 10,
renouvellements et relocations 10, données externes 15, prévision et backtest 15, qualité du notebook 10.

### Diapositive 1 — Composition contre hausse des prix (45 s) → analyse et effet de mix
- La médiane de tous les baux monte de 10,85 % en 2023 : c'est un effet de composition (deux immeubles entrent).
- Même panel d'immeubles : 0,78 %. Indice de Fisher : 1,9 %.
- Visuel : panel fixe contre tous les baux, puis la cascade (section 3 du notebook).

### Diapositive 2 — Notre définition, en une phrase (45 s) → définition
- « Croissance à unité constante du loyer effectif (net des concessions), médiane des paires, baux débutant en
  2026 (renouvellements et relocations). »
- Toujours publier à côté la hausse contractuelle, seule comparable au TAL.
- Pourquoi deux nombres : les concessions touchent 85,4 % des baux de la dernière année observée.
- Visuel : tableau comparatif des cinq définitions (section 7).

### Diapositive 3 — Même unité, bonne clé (45 s) → appariement
- Clé `prop_code + unit_code`. Montrer l'échec de `site + unit_code` : fausses paires entre appartements
  différents.
- Trois estimateurs indépendants en contre-vérification : médiane des paires, loyers répétés, effets fixes d'unité.

### Diapositive 4 — Contractuel contre effectif : les concessions (1 min) → concessions
- `PromoPay` est un forfait unique, pas un frais mensuel : `rent_effective` se reconstruit à 1,0 $ près pour
  89,7 % des baux.
- Dans le régime récent, l'effet des concessions suit l'IPC des loyers du Québec de l'année précédente.

### Diapositive 5 — Renouvellements, relocations, Québec, Ontario (1 min) → renouvellements et relocations
- Les renouvellements suivent le TAL depuis 2023 seulement ; les relocations suivent l'IPC des loyers de l'année
  précédente.
- L'immeuble d'Ottawa est exempté de la ligne directrice ontarienne (livré en 2023) : trois tests. Ses
  renouvellements ressemblent à ceux du Québec : politique de la société, pas un plafond légal.

### Diapositive 6 — Ce que dit le marché (45 s) → données externes
- IPC, SCHL, TAL, Ontario : chaque série porte sa date de publication, donc le backtest n'utilise que ce qui était
  public à la date.
- Les niveaux sont comparés à la SCHL séparément de la croissance.

### Diapositive 7 — La prévision et sa validation (1 min 30) → prévision et backtest
- Modèle par composantes et simulation de la cohorte réelle de 940 unités.
- Backtest 2023, 2024, 2025 avec données tronquées, trois millésimes, références naïves toujours affichées.
- Erreur absolue moyenne récente : 0,30 point (effectif), 0,58 point (contractuel).
- Honnêteté : bon dans le régime 2023-2025, pas universel ; les règles ont été repérées sur ces mêmes années ;
  XGBoost testé et rejeté.

### Diapositive 8 — Réponse 2026 (1 min)
- Hausse effective : 6,01 %, bande 4,74 à 7,26 %.
- Hausse contractuelle : 4,72 %, bande 3,66 à 6,22 %. Ancrage du TAL : 3,78 %.
- Scénarios (effectif) : bas 4,57 %, coin indexé 6,1 %, haut 7,57 %.
- Sensibilité si le biais récent était soustrait : 5,71 % (correction non validée).
- Message final : 2026 se joue dans la politique de concessions.

### Questions probables du jury
- Pourquoi l'effectif et pas le contractuel ? C'est le revenu encaissé ; les deux sont publiés.
- Votre backtest est-il indépendant ? Non : les règles ont été repérées sur l'historique qui sert à les tester.
- Pourquoi des poids uniformes ? Jugement explicite, la sensibilité est affichée (5,85 à 6,63 %).
- Données CRM ? Aucune dans les livrables : seuls des agrégats sont écrits, vérifiés par des tests.

### À ne pas faire
- Ne montrer aucune ligne du CRM à l'écran, seulement des agrégats.
- Ne pas présenter la bande comme une garantie de couverture.
- Ne pas présenter le calendrier du TAL comme le droit applicable à Ottawa : c'est une hypothèse de politique.
- Outils d'IA : les déclarer (ils sont listés dans `work/final/README.md`).
