# %% [markdown]
# ## 8. Données publiques et réconciliation avec la tendance interne
#
# **Ce que nous intégrons** (tout est public ; sources, licences et date de retrait dans `external/SOURCES.md`) :
# * **IPC, composante loyers** (Statistique Canada, tableau 18-10-0004-01) : Québec, Ontario, Canada, mensuel jusqu'au dernier mois publié avant la date de retrait ;
# * **SCHL, enquête sur les logements locatifs** (republiée par Statistique Canada, tableaux 34-10-0130-01 et 34-10-0133-01) : inoccupation et loyer moyen par nombre de chambres, RMR de Montréal et d'Ottawa-Gatineau (partie ontarienne) ;
# * **TAL** (estimation moyenne de l'ancienne méthode et pourcentage de base de la nouvelle) et **ligne directrice de l'Ontario** (section 6) ;
# * chiffres titres de la SCHL 2025 et mise à jour de mi-année 2026 (loyers affichés, roulement), saisis à la main avec leur URL.
#
# **Règle d'or du backtest.** Chaque ligne porte `published_on`. La prévision d'une année `t` ne peut utiliser que les valeurs publiées **au plus tard à sa date de coupure** (section 10) : le taux du TAL de l'année `t`, publié en janvier de `t`, n'est *pas* connu au 31 décembre de `t−1`.
#
# *Jeu Kaggle suggéré (« 25 000 loyers canadiens, juin 2024 »)* : non utilisé, parce qu'il exige un compte et ne couvre qu'un instantané de juin 2024 ; la SCHL fournit des niveaux par chambres avec dix ans d'historique, ce qui est plus utile à la réconciliation.

# %%
series_inventory = (external.groupby("series").agg(
    geographies=("geo", lambda geos: ", ".join(sorted(set(geos)))), first_year=("year", "min"), last_year=("year", "max"),
    rows=("value", "size"), earliest_publication=("published_on", "min"), confidence=("confidence", lambda c: ", ".join(sorted(set(c)))))
    .sort_index())
print(f"IPC mensuel publié jusqu'en {cpi_monthly.month.max():%Y-%m} (dernière publication : {cpi_monthly.published_on.max():%Y-%m-%d}).")
series_inventory

# %% [markdown]
# ### 8a. Reconstituer des loyers affichés à unité constante (série interne de plus)
# En plus des baux, les **loyers affichés** (`asking`) donnent un signal de marché propre à Équinoxe : la variation annualisée de l'affichage d'une même unité entre deux observations (au moins `asking_gap_min_years` d'écart, au plus `pair_gap_max_years`, la même fenêtre que pour les baux).

# %%
def same_unit_asking_growth(asking_table):
    ordered = asking_table.sort_values(UNIT_KEY + ["ask_date"]).copy()
    grouped = ordered.groupby(UNIT_KEY)
    ordered["prev_rent_ask"] = grouped.rent_ask.shift(1)
    ordered["prev_ask_date"] = grouped.ask_date.shift(1)
    ordered = ordered.dropna(subset=["prev_rent_ask"])
    ordered["gap_years"] = (ordered.ask_date - ordered.prev_ask_date).dt.days / CONFIG.days_per_year
    ordered = ordered[ordered.gap_years.between(CONFIG.asking_gap_min_years, CONFIG.pair_gap_max_years)]
    ordered["growth_pct"] = annualised_growth_pct(ordered.rent_ask, ordered.prev_rent_ask, ordered.gap_years)
    ordered["ask_year"] = ordered.ask_date.dt.year
    return ordered


asking_pairs = same_unit_asking_growth(asking)
asking_growth = asking_pairs.groupby("ask_year").growth_pct.median()
print("Croissance médiane des loyers affichés à unité constante :", asking_growth.round(CONFIG.table_digits).to_dict())

# %% [markdown]
# ### 8b. Réconciliation : quelle série publique explique quel segment interne, avec quel décalage ?
# Réconcilier n'est pas superposer deux courbes. Pour chaque segment interne et chaque série externe, à décalage 0 et 1 an :
# * la **corrélation** ;
# * l'**erreur absolue moyenne « laisser un an de côté »** du modèle `interne = externe + écart moyen` (l'écart est estimé *sans* l'année prédite) ;
# * `n` années.
#
# Avec sept points annuels, nous rapportons les erreurs et non des p-valeurs : **l'échantillon est trop petit pour de l'inférence formelle**, et nous le disons.

# %%
internal_series = pd.DataFrame({
    "QC renouvellements (contractuel)": qc_renewal_median,
    "QC relocations (contractuel)": pairs[(pairs.province == "Quebec") & (pairs.is_renewal == 0)].groupby("lease_year").growth_contract_pct.median(),
    "Équinoxe global (contractuel)": definition_series_all["1. même unité, contractuel"],
    "Équinoxe global (effectif)": definition_series_all["2. même unité, effectif (RETENU)"],
    "loyers affichés (même unité)": asking_growth}).loc[CONFIG.first_year:CONFIG.last_observed_year]

driver_series = pd.DataFrame({
    "TAL (année civile)": TAL_OLD,
    "IPC loyers QC (moyenne annuelle)": external_series("cpi_rent_avg_yoy", "Quebec"),
    "IPC loyers QC (déc./déc.)": external_series("cpi_rent_dec_yoy", "Quebec"),
    "SCHL Montréal : loyer moyen 2 ch. (niveau à niveau)": external_series("cmhc_avg_rent_growth_beds2", "Montreal CMA"),
}).loc[CONFIG.first_year - 1:CONFIG.last_observed_year]


def leave_one_out_mae(internal, driver):
    errors = []
    for held_out in internal.index:
        spread = (internal.drop(held_out) - driver.drop(held_out)).mean()
        errors.append(abs(internal[held_out] - (driver[held_out] + spread)))
    return float(np.mean(errors))


LOO_COLUMN = "erreur « un an de côté » (points)"
reconciliation_rows = []
for segment in internal_series:
    for driver in driver_series:
        for lag in (0, 1):
            joined = pd.concat([internal_series[segment], driver_series[driver].shift(lag)], axis=1, keys=["internal", "driver"]).dropna()
            if len(joined) < CONFIG.min_years_for_reconciliation:
                continue
            reconciliation_rows.append({
                "segment interne": segment, "série externe": driver, "décalage (ans)": lag, "n années": len(joined),
                "corrélation": joined.internal.corr(joined.driver),
                "écart moyen (interne − externe, points)": (joined.internal - joined.driver).mean(), LOO_COLUMN: leave_one_out_mae(joined.internal, joined.driver)})
reconciliation = pd.DataFrame(reconciliation_rows)
best_driver = reconciliation.sort_values(LOO_COLUMN).groupby("segment interne").head(2)
best_driver.sort_values(["segment interne", LOO_COLUMN]).round(CONFIG.table_digits)

# %%
best_by_segment = reconciliation.sort_values(LOO_COLUMN).groupby("segment interne").head(1).set_index("segment interne")
relocation_best = best_by_segment.loc["QC relocations (contractuel)"]
renewal_best = best_by_segment.loc["QC renouvellements (contractuel)"]
global_segments = ["Équinoxe global (contractuel)", "Équinoxe global (effectif)", "loyers affichés (même unité)"]
schl_level_growth = external_series("cmhc_avg_rent_growth_beds2", "Montreal CMA")[CONFIG.last_observed_year]
schl_same_sample = external_series("cmhc_montreal_avg_rent_change_2bed", "Montreal CMA")[CONFIG.last_observed_year]
narrate(f"**Lecture (à lire avec la réserve d'échantillon).**\n"
        f"* **Relocations du Québec : le lien le plus solide.** La meilleure série est « {relocation_best['série externe']} » avec un décalage de "
        f"{int(relocation_best['décalage (ans)'])} an (corrélation {fmt(relocation_best['corrélation'], CONFIG.table_digits)}, erreur « un an de côté » "
        f"{fmt(relocation_best[LOO_COLUMN])} point, la plus basse du tableau) : une relocation reprend le niveau du marché de l'année écoulée. C'est notre entrée de marché pour les relocations.\n"
        f"* **Renouvellements du Québec : l'ancrage TAL est récent.** La meilleure série atteint une erreur de {fmt(renewal_best[LOO_COLUMN], CONFIG.table_digits)} point sur toute la période, parce que "
        "les renouvellements ont largement dépassé le TAL au début de la période (section 6b). Nous l'utilisons comme **ancrage légal daté par l'avis** (section 10), pas comme une relation structurelle.\n"
        f"* **Global (contractuel et effectif) et loyers affichés :** les meilleures erreurs vont de {fmt(best_by_segment.loc[global_segments, LOO_COLUMN].min())} à "
        f"{fmt(best_by_segment.loc[global_segments, LOO_COLUMN].max())} points, parce que ces séries mélangent deux régimes (renouvellements encadrés, relocations libres) et "
        "les concessions. La prévision doit donc passer par les **composantes** (section 10).\n"
        f"* **Réserve sur la série SCHL.** Le « loyer moyen 2 chambres » de la SCHL est calculé *niveau à niveau* : il inclut les nouveaux immeubles, les conversions et la rotation de l'échantillon, "
        f"c'est-à-dire **le même effet de composition que celui montré en section 3** (ce que le chiffre à échantillon constant de la SCHL retire). Pour Montréal en {CONFIG.last_observed_year}, il donne {fmt(schl_level_growth, signed=True)} % "
        f"alors que le chiffre titre de la SCHL à échantillon constant est {fmt(schl_same_sample, signed=True)} %. Nous gardons la série niveau à niveau comme **entrée de réconciliation "
        "prudente** (elle a dix ans d'historique) et nous préférons le chiffre à échantillon constant quand il existe (une seule année).")

# %% [markdown]
# ### 8c. Niveaux : Équinoxe contre le marché
# Niveau moyen des loyers des baux débutant la dernière année observée, par nombre de chambres, contre le loyer moyen de la SCHL (octobre) pour la RMR de Montréal (Laval, Mont-Royal et Pointe-Claire en font partie) et la RMR d'Ottawa-Gatineau (The Met).
#
# **Réserve sur les niveaux.** L'extrait est un sous-ensemble publié du CRM ; rien ne garantit que ses niveaux de loyer soient ceux des annonces réelles. Les primes ci-dessous ne sont donc **qu'indicatives** ; toute la méthode du notebook repose sur des **taux de variation**, jamais sur des niveaux, pour cette raison.

# %%
level_rows = []
for geography, province_name, cmhc_geo in [("Montréal (5 immeubles du Québec)", "Quebec", "Montreal CMA"), ("Ottawa (The Met)", "Ontario", "Ottawa CMA (Ontario part)")]:
    own = leases[(leases.province == province_name) & (leases.lease_year == CONFIG.last_observed_year)]
    for beds in CONFIG.bedroom_levels:
        own_beds = own[own.beds == beds]
        cmhc_level = external_series(f"cmhc_avg_rent_beds{beds}", cmhc_geo).get(CONFIG.last_observed_year, np.nan)
        if len(own_beds) >= CONFIG.min_cell_size and not np.isnan(cmhc_level):
            level_rows.append({"marché": geography, "chambres": beds, "baux Équinoxe": len(own_beds),
                               "loyer moyen Équinoxe ($)": own_beds.rent_contract.mean(), "loyer moyen SCHL ($)": cmhc_level,
                               "prime (%)": (own_beds.rent_contract.mean() / cmhc_level - 1) * 100})
level_table = pd.DataFrame(level_rows)
display(level_table.round(CONFIG.prose_digits))
reference_series = external_series(f"cmhc_avg_rent_beds{CONFIG.reference_bedrooms}", "Montreal CMA")
premium_label = f"prime Équinoxe {CONFIG.reference_bedrooms} ch. vs SCHL Montréal (%)"
premium_by_year = pd.Series({
    year: (leases[(leases.province == "Quebec") & (leases.lease_year == year) & (leases.beds == CONFIG.reference_bedrooms)].rent_contract.mean() / reference_series.get(year, np.nan) - 1) * 100
    for year in CONFIG.backtest_years}, name=premium_label)
premium_by_year.round(CONFIG.prose_digits).to_frame()

# %%
lowest, highest = level_table.loc[level_table["prime (%)"].idxmin()], level_table.loc[level_table["prime (%)"].idxmax()]
narrate(f"**Lecture.** Équinoxe loue **au-dessus du loyer moyen du marché** : de {fmt(lowest['prime (%)'], 0, signed=True)} % ({int(lowest.chambres)} chambre(s), {lowest.marché}) "
        f"à {fmt(highest['prime (%)'], 0, signed=True)} % ({int(highest.chambres)} chambre(s), {highest.marché}). C'est attendu : la SCHL mesure l'ensemble du parc locatif "
        "(anciens immeubles compris) alors qu'Équinoxe loue des immeubles récents haut de gamme. Pour un "
        f"{CONFIG.reference_bedrooms} chambres à Montréal, la prime va de {fmt(premium_by_year.min(), 0, signed=True)} à {fmt(premium_by_year.max(), 0, signed=True)} % "
        f"de {CONFIG.backtest_years[0]} à {CONFIG.backtest_years[-1]} ; un loyer d'Équinoxe n'est donc pas « le marché ». Conséquence pour la prévision : on utilise la "
        "**variation** du marché (IPC, SCHL) comme signal, jamais son **niveau**.")

# %% [markdown]
# ### 8d. Divergences : où interne et externe se contredisent, et pourquoi
# Chaque divergence est chiffrée à partir des séries internes et des tableaux publics (aucune valeur tapée à la main).

# %%
qc_relocations = internal_series["QC relocations (contractuel)"]
cpi_rent_qc = external_series("cpi_rent_avg_yoy", "Quebec")
vacancy = {geo: external_series("cmhc_vacancy_rate", geo) for geo in ("Montreal CMA", "Ottawa CMA (Ontario part)")}
asking_index = external_series("cmhc_asking_rent_index_2bed_q4_2025", "Montreal CMA"), external_series("cmhc_asking_rent_index_2bed_q4_2025", "Ottawa CMA")
turnover = external_series("cmhc_montreal_turnover_rent_change_2bed", "Montreal CMA")[CONFIG.last_observed_year]
non_turnover = external_series("cmhc_montreal_nonturnover_rent_change_2bed", "Montreal CMA")[CONFIG.last_observed_year]
year_mid = CONFIG.backtest_years[1]
narrate(f"* **{CONFIG.last_observed_year}, relocations à {fmt(qc_relocations[CONFIG.last_observed_year], signed=True)} %** contre un loyer moyen SCHL de Montréal à "
        f"{fmt(schl_same_sample, signed=True)} % ({fmt(turnover, signed=True)} % à la relocation pour un 2 chambres, {fmt(non_turnover, signed=True)} % sans changement de locataire) : "
        "cohérent en sens, car les propriétaires du marché repartent du prix de marché à chaque relocation. Chez Équinoxe, des concessions reprennent une partie de la différence "
        f"(effectif : {fmt(effective_median[CONFIG.last_observed_year], signed=True)} %).\n"
        f"* **{year_mid}, renouvellements à {fmt(qc_renewal_median[year_mid], signed=True)} %** pendant que l'estimation du TAL valait {fmt(TAL_OLD[year_mid], signed=True)} % "
        f"et l'IPC loyers du Québec {fmt(cpi_rent_qc[year_mid], signed=True)} % : le marché général augmente plus vite que ce que la société applique à ses locataires en place.\n"
        "* **Ontario :** The Met ne suit ni la ligne directrice (exempté) ni l'IPC loyers de l'Ontario (section 6d).\n"
        f"* **Signal le plus récent :** l'indice SCHL des loyers affichés (2 chambres, base 100 au premier trimestre de 2024) vaut {fmt(asking_index[0].iloc[-1])} à Montréal et "
        f"{fmt(asking_index[1].iloc[-1])} à Ottawa au quatrième trimestre de {CONFIG.last_observed_year} (en recul à Ottawa depuis le deuxième trimestre de {CONFIG.last_observed_year} selon la SCHL), "
        f"tandis que l'inoccupation remonte (SCHL, tableau 34-10-0130, appartements et maisons en rangée : Montréal {fmt(vacancy['Montreal CMA'][year_mid])} → {fmt(vacancy['Montreal CMA'][CONFIG.last_observed_year])} % ; "
        f"Ottawa {fmt(vacancy['Ottawa CMA (Ontario part)'][year_mid])} → {fmt(vacancy['Ottawa CMA (Ontario part)'][CONFIG.last_observed_year])} %). "
        "Ces signaux servent de contexte et à l’aperçu de l’année suivante ; le modèle central conserve l’IPC de l’année précédente, règle testée en section 10.")

# %% [markdown]
# ### 8e. Les concessions suivent-elles la détente du marché ?
# On met en regard le drag moyen d'Équinoxe et l'inoccupation de la SCHL (Montréal) : la hausse des concessions de 2024-2025 est-elle une réponse à un marché plus lâche ?

# %%
vacancy_montreal = vacancy["Montreal CMA"]
concession_vs_vacancy = pd.DataFrame({
    "drag moyen Équinoxe (%)": profile_by_year.drag_pct,
    "inoccupation SCHL Montréal, année t (%)": vacancy_montreal,
    "inoccupation SCHL Montréal, année t−1 (%)": vacancy_montreal.shift(1)}).loc[CONFIG.first_year:CONFIG.last_observed_year]
display(concession_vs_vacancy.round(CONFIG.table_digits))
vacancy_correlation = {lag: concession_vs_vacancy.iloc[:, 0].corr(concession_vs_vacancy.iloc[:, lag + 1]) for lag in (0, 1)}
print("Corrélation drag × inoccupation (t, t−1) :", {lag: round(value, CONFIG.table_digits) for lag, value in vacancy_correlation.items()})

# %%
narrate(f"**Lecture.** La corrélation est faible ({fmt(vacancy_correlation[0], CONFIG.table_digits)} à l'année t, {fmt(vacancy_correlation[1], CONFIG.table_digits)} avec un an "
        f"de retard) : l'inoccupation montréalaise passe de {fmt(vacancy_montreal[CONFIG.backtest_years[0]])} à {fmt(vacancy_montreal[CONFIG.last_observed_year])} % "
        f"pendant que les concessions d'Équinoxe passent de {fmt(profile_by_year.drag_pct[CONFIG.backtest_years[0]])} à {fmt(profile_by_year.drag_pct[CONFIG.last_observed_year])} % "
        "du loyer. **Les données ne soutiennent donc pas « les concessions suivent la détente du marché »** comme explication principale ; elles ressemblent d'abord à une "
        "**politique de prix** (relever le loyer inscrit au bail et redonner la différence en concession, section 5), indexée sur l'inflation des loyers (section 10c). "
        "Cette distinction compte pour 2026 : une politique peut se maintenir ou s'inverser sans que le marché bouge. Nous gardons trois trajectoires de concessions (sections 10 et 11).")

# %%
fig, axes = plt.subplots(1, 2, figsize=STYLE.figsize_wide)
shown_years = range(CONFIG.first_year, CONFIG.last_observed_year + 1)
axes[0].plot(shown_years, internal_series.loc[shown_years, "QC relocations (contractuel)"], "o-", color=PALETTE["contract"], label="Équinoxe : relocations QC")
axes[0].plot(shown_years, internal_series.loc[shown_years, "QC renouvellements (contractuel)"], "o-", color=PALETTE["effective"], label="Équinoxe : renouvellements QC")
axes[0].plot(shown_years, driver_series.loc[shown_years, "IPC loyers QC (moyenne annuelle)"], "s--", color=PALETTE["market"], label="IPC loyers QC")
axes[0].plot(TAL_OLD.index, TAL_OLD, "k--", label="TAL, estimation moyenne")
axes[0].set(title="Interne contre externe (%)", xlim=(CONFIG.first_year - STYLE.x_margin_years, CONFIG.last_observed_year + STYLE.x_margin_years))
axes[0].legend(fontsize=STYLE.small_fontsize)
axes[1].bar(level_table.index, level_table["prime (%)"], color=PALETTE["rule"])
axes[1].set_xticks(level_table.index)
axes[1].set_xticklabels([f"{row['marché'].split(' ')[0]}\n{int(row['chambres'])} ch." for _, row in level_table.iterrows()], fontsize=STYLE.small_fontsize)
axes[1].set(title="Prime d'Équinoxe sur le loyer moyen SCHL (%)")
plt.tight_layout()
plt.show()

# %%
assert external.groupby("series").published_on.min().notna().all(), "toute série a une date de publication"
assert (external.confidence.isin(["high", "medium", "low"])).all()
assert cpi_monthly.month.max() >= year_start(CONFIG.forecast_year), "l'IPC doit couvrir l'année prévue pour l'aperçu"
assert cpi_monthly.published_on.max() <= pd.Timestamp(CONFIG.data_as_of), "aucune donnée publiée après la date de retrait"
print("Contrôles de la section 8 : OK")
