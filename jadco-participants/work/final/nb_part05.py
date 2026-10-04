# %% [markdown]
# ## 6. Renouvellements et relocations : deux régimes, deux provinces
#
# `is_renewal = 1` : un locataire en place renouvelle. `is_renewal = 0` : l'unité se libère et est relouée.
#
# **Québec (Code civil du Québec et règlement du TAL).** La hausse d'un locataire en place n'a pas de plafond : le propriétaire propose une hausse par avis ; si le locataire la refuse, le propriétaire peut demander au Tribunal administratif du logement (TAL) de fixer le loyer selon le *Règlement sur les critères de fixation de loyer*. Jusqu'en 2025, le TAL publiait chaque janvier une **estimation moyenne d'augmentation** (« ancienne méthode », ici le logement non chauffé) ; depuis 2026, le règlement fixe un **pourcentage de base** (« nouvelle méthode ») pour les avis donnés à compter du 1er janvier 2026. Pour un **nouveau locataire**, le propriétaire doit déclarer le loyer le plus bas payé dans les 12 mois précédant le début du bail (art. 1896 C.c.Q., sauf dans un immeuble visé à l'art. 1955) et le locataire peut demander au Tribunal de fixer le loyer dans les 10 jours de la conclusion du bail (deux mois du début du bail si la déclaration n'a pas été remise ; art. 1950) : la relocation est donc libre en pratique, mais pas sans recours. Enfin, **dans les cinq années qui suivent la date où l'immeuble est prêt pour l'usage**, ni le locateur ni le locataire ne peut faire fixer le loyer par le Tribunal, si le bail le prévoit (section F) et, pour un immeuble prêt depuis le 21 février 2024, s'il indique le loyer maximal des cinq ans (art. 1955) ; le locataire qui refuse la hausse doit alors quitter le logement (art. 1945) et la déclaration de l'art. 1896 ne s'applique pas. C'est le cas potentiel des propriétés mises en service récemment (test en 6c bis).
#
# **Ontario.** Un plafond annuel (la *ligne directrice*) encadre les locataires en place **sauf pour les unités occupées pour la première fois à des fins résidentielles après le 15 novembre 2018** ; la **relocation est libre** (« vacancy decontrol »). Tous les immeubles de l'extrait sont au Québec, sauf The Met (Ontario). **Nous n'appliquons jamais la règle d'une province à l'autre.**
#
# **Étape qui ajoute des données** : on charge ici les tableaux publics tidy (`external`, `cpi_monthly`), produits par `external/fetch.py` ; chaque ligne porte sa source, sa date de retrait et `published_on`, la première date où la valeur était publique. Le détail et la réconciliation sont en section 8.

# %%
external = pd.read_csv(CONFIG.external_dir / "external_annual.csv", parse_dates=["published_on"])
cpi_monthly = pd.read_csv(CONFIG.external_dir / "cpi_rent_monthly.csv", parse_dates=["month", "published_on"])


def external_series(series, geo=None):
    """Série annuelle publique indexée par année (valeur la plus récente si plusieurs lignes)."""
    rows = external[external.series == series]
    if geo is not None:
        rows = rows[rows.geo == geo]
    return rows.drop_duplicates("year", keep="last").set_index("year").value.sort_index()


TAL_OLD = external_series("tal_recommended_unheated_old_method", "Quebec")      # ancienne méthode : estimation moyenne d'augmentation, logement non chauffé
TAL_NEW_FORMAT = external_series("tal_base_percentage_new_format", "Quebec")    # nouvelle méthode : pourcentage de base (2025 recalculé, puis 2026)
ONTARIO_GUIDELINE = external_series("on_rent_guideline", "Ontario")
print("TAL, ancienne méthode (estimation moyenne, non chauffé) :", TAL_OLD.to_dict())
print("TAL, nouvelle méthode (pourcentage de base) :", TAL_NEW_FORMAT.to_dict())
print("Ontario, ligne directrice (jusqu'à l'année prévue) :", ONTARIO_GUIDELINE.loc[CONFIG.first_year:CONFIG.forecast_year].to_dict())

# %% [markdown]
# ### 6a. Croissance à unité constante : province × renouvellement
# Tableau 2 × 2 (médiane des paires, loyer contractuel). L'Ontario n'a de paires qu'à partir de 2024, car The Met ouvre en 2023.

# %%
split = (pairs[pairs.lease_year >= CONFIG.extended_backtest_years[0]]
         .assign(type=lambda d: np.where(d.is_renewal == 1, "renouvellement", "relocation"))
         .groupby(["province", "type", "lease_year"])
         .agg(pairs=("growth_contract_pct", "size"), contract=("growth_contract_pct", "median"), effective=("growth_effective_pct", "median")))
split_contract = split.unstack("lease_year")["contract"]
display(split_contract.round(CONFIG.table_digits))
display(split.unstack("lease_year")["pairs"])

# %%
qc_split = split_contract.loc["Quebec"]
on_split = split_contract.loc["Ontario"].dropna(axis=1, how="all")
renewal_led_years = [year for year in qc_split.columns if qc_split.loc["renouvellement", year] > qc_split.loc["relocation", year]]
relocation_led_years = [year for year in qc_split.columns if year not in renewal_led_years]
narrate(f"**Lecture.** Au Québec, les renouvellements mènent en {', '.join(map(str, renewal_led_years))} et les relocations en "
        f"{', '.join(map(str, relocation_led_years))} ; en {CONFIG.last_observed_year}, relocations {fmt(qc_split.loc['relocation', CONFIG.last_observed_year])} % "
        f"contre renouvellements {fmt(qc_split.loc['renouvellement', CONFIG.last_observed_year])} %. L'Ontario (The Met) suit presque exactement le Québec : "
        + " ; ".join(f"{year} : renouvellements {fmt(on_split.loc['renouvellement', year])} % (Québec {fmt(qc_split.loc['renouvellement', year])} %), "
                     f"relocations {fmt(on_split.loc['relocation', year])} % (Québec {fmt(qc_split.loc['relocation', year])} %)" for year in on_split.columns) + ".")

# %% [markdown]
# ### 6b. Mettre la règle sur le graphique
# Les renouvellements québécois sont comparés à l'**estimation moyenne du TAL** (ancienne méthode) ; ceux de l'Ontario à la ligne directrice (tracée en pointillé : non contraignante si l'immeuble est exempté, voir 6d). **Deux conventions d'année pour le TAL**, testées plutôt que supposées : (a) *année civile* (le taux publié en janvier de `t` pour l'année `t`) ; (b) *fenêtre de validité* : un taux publié en janvier de `t` couvre les baux débutant du 2 avril de `t` au 1er avril de `t+1`, donc les baux débutant du 1er janvier au 1er avril de `t` relèvent encore du taux de l'année précédente (`tal_window_first_month`, `tal_window_first_day`).

# %%
def starts_in_previous_tal_window(start_dates):
    """Vrai si le bail débute avant le premier jour de la fenêtre de validité du taux de son année : il relève du taux de l'année précédente."""
    month, day = start_dates.dt.month, start_dates.dt.day
    return (month < CONFIG.tal_window_first_month) | ((month == CONFIG.tal_window_first_month) & (day < CONFIG.tal_window_first_day))


renewals_qc = pairs[(pairs.province == "Quebec") & (pairs.is_renewal == 1)]
tal_window_share = starts_in_previous_tal_window(renewals_qc.lease_start).mean()   # mesuré dans les données, pas supposé
print(f"Part des renouvellements québécois qui relèvent du taux de l'année précédente (début avant le {CONFIG.tal_window_first_day}/{CONFIG.tal_window_first_month}) : "
      f"{tal_window_share:.1%}")

tal_window = tal_window_share * TAL_OLD.shift(1, fill_value=np.nan) + (1 - tal_window_share) * TAL_OLD
qc_renewal_median = renewals_qc.groupby("lease_year").growth_contract_pct.median()
rule_test_years = [year for year in qc_renewal_median.index if year in TAL_OLD.index and year - 1 in TAL_OLD.index]
rule_test = pd.DataFrame({
    "renouvellements QC (médiane)": qc_renewal_median.loc[rule_test_years],
    "TAL année civile": TAL_OLD.loc[rule_test_years],
    "TAL fenêtre de validité": tal_window.loc[rule_test_years]})
rule_test["écart (année civile)"] = rule_test.iloc[:, 0] - rule_test.iloc[:, 1]
rule_test["écart (fenêtre)"] = rule_test.iloc[:, 0] - rule_test.iloc[:, 2]
rule_test_mae = pd.DataFrame({
    f"{min(years)}-{max(years)}": {"année civile": rule_test.loc[years, "écart (année civile)"].abs().mean(), "fenêtre": rule_test.loc[years, "écart (fenêtre)"].abs().mean()}
    for years in (rule_test_years, [year for year in rule_test_years if year >= CONFIG.backtest_years[0]])}).T
display(rule_test_mae.round(CONFIG.table_digits))
rule_test.round(CONFIG.table_digits)

# %%
first_rule_year = rule_test_years[0]
recent_rule_years = [year for year in rule_test_years if year >= CONFIG.backtest_years[0]]
narrate(f"**Lecture.** En {first_rule_year}, les renouvellements ({fmt(rule_test.loc[first_rule_year].iloc[0])} %) dépassent de loin l'estimation du TAL "
        f"({fmt(rule_test.loc[first_rule_year].iloc[1])} %) ; à partir de {recent_rule_years[0]}, l'écart reste sous "
        f"{fmt(rule_test.loc[recent_rule_years, 'écart (année civile)'].abs().max())} point. Seuls {fmt(tal_window_share * 100, 0)} % des renouvellements "
        f"débutent avant la fenêtre de validité de l'année, donc les deux conventions donnent des erreurs du même ordre ({fmt(rule_test_mae.iloc[-1, 0], CONFIG.table_digits)} contre "
        f"{fmt(rule_test_mae.iloc[-1, 1], CONFIG.table_digits)} point sur {rule_test_mae.index[-1]}). **La prévision applique la fenêtre de validité** : "
        "c'est la règle de droit, et c'est elle qui permet de dater correctement le changement de méthode de 2026 (section 10d). "
        "L'ancrage TAL est utile pour les renouvellements récents, **pas une loi de la nature** : il n'explique pas les années antérieures.")

# %%
fig, axes = plt.subplots(2, 2, figsize=STYLE.figsize_grid, sharey=True)
panels = [("Quebec", 1, "Québec · renouvellements", TAL_OLD), ("Quebec", 0, "Québec · relocations", None),
          ("Ontario", 1, "Ontario (The Met) · renouvellements", ONTARIO_GUIDELINE), ("Ontario", 0, "Ontario (The Met) · relocations", None)]
for ax, (province, renewal_flag, title, rule) in zip(axes.ravel(), panels):
    subset = pairs[(pairs.province == province) & (pairs.is_renewal == renewal_flag) & (pairs.lease_year >= CONFIG.first_year)]
    by_year = subset.groupby("lease_year").agg(contract=("growth_contract_pct", "median"), effective=("growth_effective_pct", "median"))
    ax.plot(by_year.index, by_year.contract, "o-", color=PALETTE["contract"], label="contractuel")
    ax.plot(by_year.index, by_year.effective, "o-", color=PALETTE["effective"], label="effectif")
    if rule is not None:
        shown = rule.loc[by_year.index.min():by_year.index.max()]
        ax.plot(shown.index, shown, "k--", label="TAL, estimation moyenne (ancienne méthode)" if province == "Quebec" else "ligne directrice (non contraignante si exempté)")
    ax.set(title=title)
    ax.set_xticks(range(CONFIG.first_year, CONFIG.last_observed_year + 1))
    ax.legend(fontsize=STYLE.small_fontsize)
fig.supylabel("hausse annualisée à unité constante (%)")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 6c. Part de renouvellements (poids du mélange)
# Le poids des renouvellements dans la cohorte sert de pondération à la prévision (section 10) ; vérifions qu'il est stable.

# %%
renewal_share = pairs.groupby(["province", "lease_year"]).is_renewal.mean().unstack("province").loc[CONFIG.first_year:] * 100
renewal_share.round(CONFIG.prose_digits)

# %%
qc_share = renewal_share["Quebec"]
narrate(f"**Lecture.** La part de renouvellements au Québec reste entre {fmt(qc_share.min(), 0)} et {fmt(qc_share.max(), 0)} % depuis {CONFIG.first_year} "
        f"(écart-type {fmt(qc_share.std())} point) : on peut l'estimer sur les dernières années sans craindre une dérive. "
        "Sa variabilité d'une année à l'autre fait partie de l'erreur mesurée par le backtest (section 10d).")

# %% [markdown]
# ### 6c bis. Article 1955 : les immeubles de cinq ans ou moins renouvellent-ils autrement ?
# Si la société profitait de l'exemption de l'art. 1955 (pas de fixation par le Tribunal dans un immeuble récent), les renouvellements des propriétés récentes devraient être plus élevés que ceux des propriétés plus anciennes la même année. **Approximation assumée :** la loi compte cinq ans à partir de la date où l'immeuble est prêt pour l'usage, que l'extrait ne donne pas ; nous comptons les années civiles depuis le premier bail de la propriété dans l'extrait (`entry_year`, section 3), et nous ne savons pas si les baux contiennent la mention de la section F.

# %%
qc_renewal_pairs = renewals_qc[renewals_qc.lease_year >= CONFIG.backtest_years[0]].assign(
    property_age=lambda d: d.lease_year - d.prop_code.map(entry_year))
qc_renewal_pairs["recent_property"] = qc_renewal_pairs.property_age <= CONFIG.new_building_age_years
age_test = qc_renewal_pairs.groupby(["lease_year", "recent_property"]).growth_contract_pct.agg(["median", "size"]).unstack("recent_property")
age_test.columns = [f"{statistic} ({'récente' if recent else 'ancienne'})" for statistic, recent in age_test.columns]
age_test["écart récente − ancienne (points)"] = age_test["median (récente)"] - age_test["median (ancienne)"]
age_test.round(CONFIG.table_digits)

# %%
narrate(f"**Lecture.** L'écart entre propriétés récentes et anciennes reste entre {fmt(age_test.iloc[:, -1].min(), CONFIG.table_digits, signed=True)} et "
        f"{fmt(age_test.iloc[:, -1].max(), CONFIG.table_digits, signed=True)} point selon l'année : **la société applique une seule politique de renouvellement**, "
        "que l'immeuble puisse ou non échapper à la fixation par le Tribunal. C'est le même constat qu'à Ottawa (6d) et c'est ce qui justifie, en section 10, "
        "un ancrage de renouvellement commun à toutes les propriétés.")

# %% [markdown]
# ### 6d. L'immeuble d'Ottawa selon les règles de l'Ontario (bonus)
# **Constat qui pourrait passer pour une erreur.** Les renouvellements de The Met dépassent nettement la ligne directrice ontarienne (chiffres ci-dessous). Un propriétaire soumis au plafond ne pourrait pas la dépasser sans ordonnance. Ce n'est pas une erreur des données : **The Met est exempté** (tour résidentielle livrée en 2023 derrière une façade patrimoniale ; premier bail de l'extrait en 2023, donc usage résidentiel pour la première fois après le 15 novembre 2018 : sources en section 13 et dans `external/SOURCES.md` ; hypothèse à confirmer auprès du propriétaire, d'où le régime plafonné gardé en contrefactuel). On le démontre par trois tests.

# %%
met_pairs = pairs[pairs.prop_code == "metcalfe"]
met_renewals = met_pairs[met_pairs.is_renewal == 1].copy()
met_renewals["guideline_pct"] = met_renewals.lease_year.map(ONTARIO_GUIDELINE)
exemption_cutoff = pd.Timestamp(CONFIG.ontario_exemption_cutoff)
first_lease_met = leases[leases.prop_code == "metcalfe"].lease_start.min()
min_months_between = CONFIG.ontario_min_months_between_increases - CONFIG.month_tolerance_between_increases
regime_test = met_renewals.groupby("lease_year").apply(lambda g: pd.Series({
    "renouvellements": len(g), "médiane (%)": g.growth_contract_pct.median(), "ligne directrice (%)": g.guideline_pct.iloc[0],
    "part au-dessus de la ligne directrice": (g.growth_contract_pct > g.guideline_pct).mean(),
    f"part ≥ {CONFIG.ontario_min_months_between_increases} mois entre deux hausses": ((g.lease_start - g.prev_lease_start).dt.days / CONFIG.days_per_year * CONFIG.months_per_year >= min_months_between).mean()}),
    include_groups=False)
print(f"Test 1, date d'occupation : premier bail de The Met = {first_lease_met:%Y-%m-%d} ; seuil d'exemption = {exemption_cutoff:%Y-%m-%d} → "
      f"{'EXEMPTÉ' if first_lease_met > exemption_cutoff else 'ASSUJETTI'}")
display(regime_test.round(CONFIG.table_digits))
met_vs_cpi = pd.DataFrame({"The Met, renouvellements (%)": regime_test["médiane (%)"],
                           "IPC loyers Ontario (moyenne annuelle, %)": external_series("cpi_rent_avg_yoy", "Ontario").loc[regime_test.index],
                           "ligne directrice (%)": regime_test["ligne directrice (%)"]})
met_vs_cpi.round(CONFIG.table_digits)

# %%
spacing_column = regime_test.columns[-1]
narrate(f"**Lecture des trois tests.**\n"
        f"1. **Date d'occupation :** premier bail le {first_lease_met:%Y-%m-%d}, postérieur au {exemption_cutoff:%Y-%m-%d} → la ligne directrice **ne s'applique pas**. "
        f"Les hausses de renouvellement ({', '.join(fmt(value) + ' %' for value in regime_test['médiane (%)'])}) sont donc légales.\n"
        f"2. **Comportement :** {', '.join(fmt(share * 100, 0) + ' %' for share in regime_test['part au-dessus de la ligne directrice'])} des renouvellements de The Met "
        f"dépassent la ligne directrice ({', '.join(map(str, regime_test.index))}) : le plafond ne lie pas. Ils sont **presque identiques aux renouvellements québécois** "
        f"({', '.join(fmt(qc_split.loc['renouvellement', year]) + ' %' for year in regime_test.index)}) et ne suivent **ni** la ligne directrice **ni** l'IPC loyers "
        f"de l'Ontario ({', '.join(fmt(value) + ' %' for value in met_vs_cpi.iloc[:, 1])}) : la politique d'augmentation est celle de la société, appliquée à un immeuble exempté.\n"
        f"3. **Règle des {CONFIG.ontario_min_months_between_increases} mois :** {', '.join(fmt(share * 100, 0) + ' %' for share in regime_test[spacing_column])} des hausses "
        "d'une même unité sont espacées d'au moins douze mois, à un mois près (la règle est respectée en pratique ; l'avis de 90 jours n'est pas observable dans les données).\n\n"
        f"**Conséquence pour la prévision {CONFIG.forecast_year}.** Régime central : *exempté*. The Met reçoit la **politique de la société** (l'ancrage de renouvellement commun, 6c bis), "
        "parce que c'est ce que ses données montrent ; **aucune règle du Québec n'est appliquée à Ottawa en droit**, et nous n'avons pas d'entrée de marché propre à Ottawa "
        f"pour ses relocations (limite assumée, section 12). Régime contrefactuel : renouvellements plafonnés à la ligne directrice {CONFIG.forecast_year} "
        f"({fmt(ONTARIO_GUIDELINE[CONFIG.forecast_year])} %). L'écart entre les deux est la **valeur de l'exemption** (section 11d). "
        "**La ligne directrice de l'Ontario n'est jamais appliquée aux cinq immeubles du Québec.**")

# %%
# Contrôles de validité ----------------------------------------------------------------------------------------
assert first_lease_met > exemption_cutoff, "The Met doit être postérieur au seuil d'exemption"
assert set(pairs[pairs.province == "Ontario"].prop_code) == {"metcalfe"}, "seul The Met est en Ontario"
assert (pairs.groupby("prop_code").province.nunique() == 1).all(), "une province par code de propriété"
print("Contrôles de la section 6 : OK")
