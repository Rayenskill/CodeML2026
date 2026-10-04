# Collection Équinoxe — hausse de loyer 2026 : plan de présentation (8 minutes, 8 diapositives)

*Fil conducteur du défi : Définir → Mesurer → Tester → Défendre. Les chiffres viennent de `outputs/final_answer.json` ; ne pas les retaper.*

## 1. Composition contre hausse des prix (45 s)
- La médiane des loyers de tous les baux monte de 10,8 % en 2023 : c'est la **composition** (Le Carlyle et The Met entrent).
- Même panel d'immeubles : 0,8 %. Indice de Fisher : 1,9 %. Décomposition exacte : l'entrée de nouveaux immeubles explique l'essentiel.
- *Visuel : section 3 du notebook (panel fixe contre tous les baux, puis la cascade).*

## 2. Notre définition, en une phrase (45 s)
- **Croissance à unité constante du loyer effectif (net des concessions), médiane des paires, baux débutant en 2026 (renouvellements + relocations)**
- Toujours publiée à côté : la hausse **contractuelle** (seule comparable au TAL). Pourquoi deux nombres : les concessions (85,4 % des baux de la dernière année observée) séparent le loyer au bail du revenu encaissé.
- *Visuel : section 7, le tableau comparatif des cinq définitions.*

## 3. Même unité, bonne clé (45 s)
- Clé `prop_code + unit_code`. Démonstration de l'échec de `site + unit_code` : fausses paires entre appartements différents, dispersion accrue, vraies paires détruites.
- Trois estimateurs indépendants (médiane des paires, loyers répétés, effets fixes d'unité) servent de contre-vérification ; leurs écarts sont chiffrés dans le notebook.

## 4. Contractuel contre effectif : l'histoire des concessions (1 min)
- `PromoPay` est un forfait unique, pas un frais mensuel : `rent_effective` se reconstruit à ≤ 1,0 $ pour 89,7 % des baux.
- Dans le régime récent, le drag annuel moyen suit **l'IPC loyers du Québec de l'année précédente**, sans paramètre. C'est ce qui explique l'écart de 2024-2025.

## 5. Renouvellements, relocations, Québec, Ontario (1 min)
- Les renouvellements suivent le TAL depuis 2023 seulement ; avant, non. Les relocations suivent l'IPC loyers de l'année précédente.
- The Met (Ottawa) est **exempté** de la ligne directrice (immeuble livré en 2023) : trois tests. Ses renouvellements ressemblent à ceux du Québec : politique de la société, pas un plafond. Valeur de l'exemption chiffrée.

## 6. Ce que dit le marché (45 s)
- IPC, SCHL, TAL, Ontario : chaque série porte sa date de publication (`published_on`), donc le backtest n'utilise que ce qui était public.
- Les relocations et concessions suivent récemment l'IPC loyers de l'année précédente. Le TAL est un ancrage de politique récent, avec un cadre juridique propre au Québec. Les niveaux sont comparés séparément à la SCHL, sans confondre cette comparaison avec la croissance.

## 7. La prévision et sa validation (1 min 30)
- Modèle par composantes + simulation de la cohorte réelle de 940 unités ; backtest 2023-2024-2025 avec données tronquées et sources datées, trois millésimes, références naïves toujours affichées.
- Erreur absolue moyenne récente : 0,3 point (effectif), 0,6 point (contractuel).
- Honnêteté : bon dans le régime 2023-2025, **pas universel** (2021-2022) ; les règles ont été repérées sur ces années ; XGBoost testé et **rejeté**.

## 8. Réponse 2026 (1 min)
- **Hausse effective : 6,0 %** [bande 10-90 % : 4,7 à 7,3 %], modèle structurel (TAL daté par l'avis, IPC loyers de l'année précédente, concessions à deux marges) ; grille des inconnues de 2026 pondérée uniformément ; bande = grille + échantillonnage + erreur de modèle du backtest.
- **Hausse contractuelle : 4,7 %** [3,7 à 6,2 %]. Ancrage du TAL daté par l'avis : 3,78 %.
- Scénarios (effectif) :
  - bas (cliquet) : 4,6 %
  - coin indexé + pourcentage de base : 6,1 %
  - haut (ancienne méthode) : 7,6 %
- Les coins de politique sont des scénarios, pas des observations. Le calendrier TAL est un proxy de politique pour The Met ; ce n’est pas le droit applicable à Ottawa.
- Sensibilité au biais récent : 5,7 % si le biais historique était soustrait ; cette correction n’est pas validée.
- **Message final : 2026 se joue dans la politique de concessions.** Chaque point de drag en plus retire environ un point à la hausse effective (chiffré dans la section 10g du notebook).

---
### Questions probables du jury
- *Pourquoi l'effectif et pas le contractuel ?* Le revenu encaissé ; les deux sont publiés.
- *Votre backtest est-il indépendant ?* Les règles ont été repérées sur l'historique qui sert aussi à les tester ; nous n'en faisons pas une preuve de performance future.
- *Pourquoi des poids uniformes ?* Jugement explicite faute de preuve suffisante ; la sensibilité aux autres lectures est affichée.
- *Pourquoi conserver le biais récent ?* Il est montré et inclus dans l'erreur quadratique ; le soustraire à partir des mêmes quelques années ne serait pas une correction indépendante.
- *Et si l'indexation des concessions s'arrête ?* Scénario cliquet ; le drag n'a jamais baissé dans les données, mais le moteur IPC est déjà passé sous le drag précédent au début de l’historique.
- *Données CRM ?* Aucune dans les livrables : seuls des agrégats sont écrits (tests automatisés).
