# %% [markdown]
# ## 12. Limites et hypothèses (énoncées, pas cachées)
#
# **Hypothèses de la prévision**
# 1. **Définition** : hausse à unité constante du loyer *effectif*, médiane des paires annualisées, baux débutant en 2026 ; le contractuel est publié à côté.
# 2. **Cohorte** : les unités dont le dernier bail connu se termine de sorte que le suivant débute en 2026 (le nouveau bail commence le lendemain de la fin de l'ancien). La couverture de cette hypothèse est mesurée ci-dessous sur les années passées.
# 3. **Renouvellements** : le taux du TAL applicable à chaque unité, selon la fenêtre de validité du taux et la date légale de l'avis (sections 6b et 10d). C'est la loi ; la société pourrait s'en écarter, d'où les deux lectures extrêmes de la grille.
# 4. **Relocations** : variation de l'IPC loyers du Québec de l'année précédente, sans paramètre ajusté (sections 8b et 10c).
# 5. **Concessions** : le niveau du drag suit l'IPC loyers du Québec de l'année précédente ; sa forme suit deux marges (pénétration `1 − exp(−k × drag)`, `k` estimé à la date de coupure ; profondeur tirée de la distribution de chaque propriété). **Le drag n'a jamais baissé dans les données** : le cliquet et le demi-cliquet restent dans la grille.
# 6. **Dispersion** : les écarts entre paires d'un même type ressemblent à ceux de l'année précédente (résidus rééchantillonnés).
# 7. **Ontario (The Met)** : exempté de la ligne directrice (immeuble livré en 2023, sources publiques) ; suit la politique de renouvellement de la société (sections 6c bis et 6d). Régime plafonné présenté en contrefactuel.
# 8. **Information** : la prévision en direct utilise les données publiques à la date de retrait (millésime `october`) ; le backtest rejoue chaque millésime avec les seules valeurs publiées à la date de coupure.
# 9. **A priori** : les lectures des deux inconnues de 2026 sont pondérées uniformément, faute de preuve suffisante pour les départager (section 10d ; précédent historique affiché) ; la section 11b montre la réponse sous les seules règles observées, et la section 10f bis la sensibilité à chaque réglage déclaré.
#
# **Couverture de la cohorte.** Les unités d'une cohorte sont-elles réellement relouées dans l'année ? On le vérifie sur les années du backtest, avec la cohorte telle qu'elle était connue à chaque date de coupure.

# %%
coverage_rows = []
for target_year in CONFIG.backtest_years:
    cohort_year = build_cohort(prepare_context(*known_at(leases, asking, target_year)), target_year)
    started = set(zip(leases[leases.lease_year == target_year].prop_code, leases[leases.lease_year == target_year].unit_code))
    covered = np.array([(code, unit) in started for code, unit in zip(cohort_year.prop_code, cohort_year.unit_code)])
    coverage_rows.append({"année": target_year, "unités de la cohorte": len(cohort_year), "relouées dans l'année (%)": covered.mean() * 100,
                          "non relouées à stelz1": int((~covered & (cohort_year.prop_code == "stelz1").to_numpy()).sum()), "non relouées au total": int((~covered).sum())})
coverage = pd.DataFrame(coverage_rows).set_index("année")
display(coverage.round(CONFIG.prose_digits))
stelz1_leases = leases[leases.prop_code == "stelz1"].groupby("lease_year").size()
stelz1_in_cohort = int((mixture["cohort"].prop_code == "stelz1").sum())
print(f"Cohorte {CONFIG.forecast_year} : {mixture['n_units']} unités, dont {stelz1_in_cohort} à stelz1 ; baux de stelz1 par année :", stelz1_leases.loc[CONFIG.backtest_years[0]:].to_dict())

# %%
early_contract_errors = structural_live_rows.loc[early_years, "error_contract"]
relet_share = coverage["relouées dans l'année (%)"]
lowest_year = relet_share.idxmin()
narrate(f"**Limites (chiffrées).**\n"
        f"* **Couverture.** Les cohortes passées ont été relouées dans l'année à {fmt(relet_share.min(), 0)}-"
        f"{fmt(relet_share.max(), 0)} % ; en {lowest_year}, l'année la plus basse, {coverage.loc[lowest_year, 'non relouées à stelz1']} des "
        f"{coverage.loc[lowest_year, 'non relouées au total']} unités non relouées sont à `stelz1`, qui se vide "
        f"({' → '.join(str(int(count)) for count in stelz1_leases.loc[CONFIG.backtest_years[0]:])} baux par an) pendant que `stelz3` ouvre en réutilisant ses codes d'unité. "
        f"La cohorte {CONFIG.forecast_year} compte {stelz1_in_cohort} unités de `stelz1` ({fmt(stelz1_in_cohort / mixture['n_units'] * 100)} % de la cohorte), qui ne seront "
        "probablement pas relouées : la cohorte surestime d'autant le nombre de paires.\n"
        f"* **Règles repérées sur les années qu'elles prédisent.** L'indexation des concessions et des relocations sur l'IPC a été trouvée en regardant 2018-2025 : le backtest "
        "la favorise et la bande d'erreur de modèle est optimiste. Huit points annuels ne prouvent pas une loi ; la prévision dépend de sa poursuite.\n"
        f"* **Peu de régimes.** Hors du régime d'indexation ({', '.join(map(str, early_years))}), l'erreur contractuelle du modèle atteint "
        f"{', '.join(fmt(value, signed=True) for value in early_contract_errors)} points : les renouvellements dépassaient alors largement le TAL.\n"
        f"* **Ottawa : peu de recul.** The Met n'a que {met_pairs.lease_year.nunique()} années de paires ; ses estimations sont plus incertaines que celles du Québec.")

# %% [markdown]
# **Autres limites**
# * **Valeurs du TAL.** Les estimations de l'ancienne méthode de 2022 à 2024 sont lues dans le texte des communiqués officiels du TAL ; la valeur 2025 est lue dans l'image du tableau 2 du communiqué officiel (le texte n'est pas extractible) et 2021 dans le communiqué du TAL du 20 janvier 2021 diffusé par CNW (voir `external/manual_public_rates.csv`) ; le taux « ancienne méthode » de la fenêtre 2026 n'est pas publié : il est prévu par la formule de la section 10c ; les pourcentages de base de la nouvelle méthode viennent des sites officiels et sont retrouvés par leur formule (section 10c). Les valeurs antérieures à 2021, publiées sous forme d'images, ne sont pas utilisées.
# * **Extrait publié, pas le portefeuille.** Les volumes ne sont pas représentatifs : l'occupation, l'inoccupation, l'absorption et le rôle de loyers du portefeuille **ne sont pas calculés** (le défi les exclut). L'inoccupation citée vient de la SCHL (publique). Les **niveaux** de loyer de l'extrait ne sont jamais comparés à des annonces réelles : seuls des taux de variation servent à la méthode.
# * **Pas de structure exploitable au niveau de l'unité.** Ni XGBoost (section 10h) ni le test de persistance des segments (section 11e) ne trouvent de signal propre à l'unité ou au segment qui survive hors période : les données se réduisent à quelques paramètres annuels (taux de renouvellement, hausse des relocations, niveau et fréquence des concessions), dont trois sont liés à des séries publiques.
# * **Pas d'inférence formelle** sur les réconciliations (peu de points annuels) : nous rapportons des erreurs et des corrélations, pas des p-valeurs.
# * **Le fichier `listings`** ne contient aucun prix 2026 (section 10g) ; nous ne le traitons pas comme tel.
#
# **Ce que nous ferions avec plus de données** : les concessions par bail de 2026 (pour mesurer directement la politique), un historique d'affichages hebdomadaire, les dates réelles des avis de renouvellement (pour remplacer l'hypothèse d'avis uniforme dans la fenêtre légale), les dates de non-renouvellement et les unités vacantes (pour une vraie probabilité de renouvellement par immeuble).

# %% [markdown]
# ## 13. Références
#
# ### Sources publiques (détail, licences et dates de retrait dans `external/SOURCES.md` ; valeurs dans `external/tidy/`)
# * **Statistique Canada**, tableau 18-10-0004-01, *Indice des prix à la consommation*, composante loyers et ensemble, Québec, Ontario, Canada : https://www150.statcan.gc.ca/t1/tbl1/fr/tv.action?pid=1810000401 (Licence du gouvernement ouvert – Canada).
# * **SCHL**, *Enquête sur les logements locatifs* (taux d'inoccupation et loyers moyens par nombre de chambres, RMR de Montréal et d'Ottawa-Gatineau, partie ontarienne), republiée par Statistique Canada, tableaux 34-10-0130-01 et 34-10-0133-01 : https://www150.statcan.gc.ca/t1/tbl1/fr/tv.action?pid=3410013001 et https://www150.statcan.gc.ca/t1/tbl1/fr/tv.action?pid=3410013301.
# * **SCHL**, *Rapport sur le marché locatif 2025* (chiffres titres pour Montréal : loyer moyen à échantillon constant, hausse à la relocation et sans changement de locataire) et *Mise à jour de mi-année 2026* (indice des loyers affichés) : https://www.cmhc-schl.gc.ca/professionals/housing-markets-data-and-research/market-reports/rental-market-reports-major-centres et https://www.cmhc-schl.gc.ca/observer/2026/2026-mid-year-rental-market-update.
# * **Tribunal administratif du logement**, communiqués annuels sur le calcul de l'ajustement des loyers (2022 à 2025, estimation moyenne de l'ancienne méthode) : https://www.tal.gouv.qc.ca/ ; pourcentage de base 2026 : https://www.quebec.ca/nouvelles/actualites/details/le-calcul-de-lajustement-des-loyers-en-2026-68053 ; page des pourcentages : https://www.tal.gouv.qc.ca/fr/reconduction-du-bail-et-fixation-de-loyer/pourcentages-applicables-aux-criteres-de-fixation-de-loyer.
# * **Gouvernement du Québec**, décret 1455-2025 (*Règlement modifiant le Règlement sur les critères de fixation de loyer*, nouvelle méthode et art. `3.1` : formule du pourcentage de base ; art. 6 et 8 : disposition transitoire pour les avis donnés à compter du 1er janvier 2026), *Gazette officielle du Québec* : https://www.publicationsduquebec.gouv.qc.ca/fileadmin/gazette/pdf_encrypte/lois_reglements/2025F/86904.pdf.
# * **Code civil du Québec**, art. 1896 (déclaration du loyer le plus bas des 12 derniers mois), 1942 (délais de l'avis de modification), 1950 (fixation à la demande du nouveau locataire) et 1955 (immeubles de cinq ans ou moins) : https://www.legisquebec.gouv.qc.ca/fr/document/lc/CCQ-1991.
# * **Ontario**, *Residential rent increases* (ligne directrice annuelle ; exemption des unités occupées pour la première fois après le 15 novembre 2018 ; règle des 12 mois) : https://www.ontario.ca/page/rent-increase-guideline.
# * **The Met, 180 rue Metcalfe, Ottawa** (tour résidentielle de 303 logements, occupation à partir de mai 2023, derrière la façade patrimoniale du Medical Arts Building) : RENX, *Jadco doubles down on Ottawa with 2nd downtown multires tower* (19 juin 2023, https://renx.ca/jadco-doubles-down-on-ottawa-with-2nd-downtown-multires-tower) ; Ottawa Business Journal (28 septembre 2018, https://obj.ca/?p=77759) ; contexte du projet initial (promoteur Toth Equity) : Patrimoine Ottawa (https://heritageottawa.org/node/1115).
# * **Jeu Kaggle suggéré** (loyers canadiens, juin 2024) : consulté comme piste, **non utilisé** (compte requis, instantané unique).
#
# ### Méthodes
# * Bailey, M. J., Muth, R. F. et Nourse, H. O. (1963). *A regression method for real estate price index construction*. Journal of the American Statistical Association.
# * Case, K. E. et Shiller, R. J. (1987). *Prices of single-family homes since 1970: new indexes for four cities*. New England Economic Review (pondération par la variance prédite selon l'écart).
# * Décomposition *shift-share* (effet de prix à l'intérieur des segments contre effet de composition) et indice de Fisher (moyenne géométrique des indices de Laspeyres et de Paasche).
# * Simulation Monte-Carlo à nombres aléatoires communs (les écarts entre scénarios ne viennent que des entrées) ; validation à origine glissante avec données publiques datées (*point-in-time*).
# * Chen, T. et Guestrin, C. (2016). *XGBoost: a scalable tree boosting system*. KDD ; Lundberg, S. et Lee, S.-I. (2017). *A unified approach to interpreting model predictions* (SHAP).
#
# ### Outils d'intelligence artificielle utilisés (déclaration exigée par le défi)
# * **Claude Code (Anthropic, modèle `Claude Sonnet 5.5`)** : assistant de programmation et de rédaction. Il a servi à écrire et tester le code de ce notebook, à explorer les données, à rechercher et lire les sources publiques ci-dessus et à rédiger les explications.
# * Une **seconde session Claude Code (modèle `Claude Opus 5.5`)** a mené en parallèle des analyses indépendantes (simulation de cohorte, XGBoost, loyers répétés, modèle hédonique), des relectures et la refonte finale du notebook (paramètres estimés ou déclarés, texte chiffré rédigé par le code, ancrage du TAL daté par l'avis, modèle des concessions à deux marges). Ses apports retenus : la **découverte de l'indexation des concessions et des relocations sur l'IPC loyers de l'année précédente** et la **reconstruction des deux formules du TAL** ; nous avons vérifié chaque résultat sur les données avant de l'intégrer.
# * **Codex (OpenAI)** : nettoyage du code, contrôles des entrées, reconstruction et vérification de la livraison, graphiques et présentation.
# * **Chaque chiffre du notebook est produit par le code exécuté ici** ; les valeurs publiques saisies à la main (TAL, ligne directrice, chiffres titres de la SCHL) sont dans `external/manual_public_rates.csv` avec leur URL, leur date de publication, leur date de retrait et un niveau de confiance.
# * Codex (OpenAI) a repris le travail interrompu : reconstruction, contrôles, corrections de cohérence et sensibilité au biais récent.
# * L'équipe a relu et validé les choix de méthode ; les décisions de jugement (définition, hypothèses, a priori, lecture des règles) sont les siennes.

# %%
print("Notebook terminé : toutes les sections ont été exécutées et tous les contrôles de validité sont passés.")
