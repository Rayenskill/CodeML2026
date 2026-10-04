# %% [markdown]
# ## 11. La prévision 2026 : réponse, scénarios, intervalles, immeuble × chambres
#
# ### 11a. Estimation selon chaque définition
# Prévision de la cohorte réelle de 2026 (les unités dont le bail échoit en 2026) par le modèle structurel calculé en section 10i, avec l'information publique à la date de retrait (millésime `october`). **La définition principale est l'effectif, médiane des paires.** Les valeurs sont les médianes de la distribution du modèle (a priori uniforme sur les lectures de la section 10d) ; l'incertitude est en 11c.

# %%
pooled, pooled_weights = mixture["pooled"], mixture["weights"]


def mixture_percentile(statistic, percentile, source=None):
    """Percentile pondéré d'une statistique du modèle structurel (tirages de toutes les combinaisons, pondérés par leurs a priori)."""
    result = source or mixture
    return weighted_percentile(result["pooled"][statistic], result["weights"], percentile)


expected_renewal_share = renewal_probability(mixture["context"], CONFIG.forecast_year, mixture["cohort"]).mean()
definitions_2026 = pd.DataFrame({
    f"{CONFIG.last_observed_year} réalisé": [definition_series_all[name][CONFIG.last_observed_year] for name in DEFINITIONS],
    f"{CONFIG.forecast_year} prévu (médiane du modèle)": [mixture_percentile(statistic, MEDIAN_PERCENTILE) for statistic in ("contract", "effective", "renewal_contract", "relocation_contract")]},
    index=list(DEFINITIONS))
definitions_2026["variation (points)"] = definitions_2026.iloc[:, 1] - definitions_2026.iloc[:, 0]
print(f"Cohorte {CONFIG.forecast_year} : {mixture['n_units']} unités ; part de renouvellement attendue : {expected_renewal_share:.1%}")
definitions_2026.round(CONFIG.table_digits)

# %%
cohort_live = mixture["cohort"]
start_quarter = cohort_live.next_start.dt.quarter.to_numpy()
notice_timing = pd.DataFrame({"unités": pd.Series(start_quarter).value_counts().sort_index(),
                              "part de la fenêtre d'avis avant la réforme": pd.Series(mixture["share_old"]).groupby(start_quarter).mean(),
                              **{f"ancrage « {name} » (%)": pd.Series(values).groupby(start_quarter).mean() for name, values in mixture["readings"].items()}})
notice_timing.index.name = f"trimestre de début du bail {CONFIG.forecast_year}"
display(notice_timing.round(CONFIG.table_digits))
law_anchor = mixture["readings"][LAW_READING]
narrate(f"**Lecture.** La cohorte compte {mixture['n_units']} unités ; {fmt(expected_renewal_share * 100, 0)} % devraient renouveler. Les renouvellements des définitions 3 et 4 "
        f"se lisent face au droit : l'ancrage **daté par l'avis** vaut en moyenne {fmt(law_anchor.mean(), CONFIG.table_digits)} % sur la cohorte, de "
        f"{fmt(notice_timing.iloc[0, -2], CONFIG.table_digits)} % pour les baux du premier trimestre (avis donnés en {CONFIG.last_observed_year}, ancienne méthode) à "
        f"{fmt(notice_timing.iloc[-1, -2], CONFIG.table_digits)} % pour ceux du dernier trimestre (pourcentage de base). La définition 5 (médiane du portefeuille) n'apparaît pas : "
        "elle dépend du niveau des loyers de nouveaux immeubles qu'on ne peut pas prévoir à partir de la cohorte, et elle mesure le mix (section 3).")

# %% [markdown]
# ### 11b. La grille des scénarios, et ce que pèse chaque jugement
# Chaque case est une simulation de la cohorte (mêmes nombres aléatoires partout) : lignes = lecture de renouvellement, colonnes = trajectoire des concessions (section 10d). Trois coins sont nommés : **bas** (pourcentage de base pour tous, cliquet), **coin indexé** (pourcentage de base pour tous, concessions indexées) et **haut** (ancienne méthode pour tous, concessions indexées). Le tableau des a priori montre ensuite la réponse sous d'autres poids : si elle bougeait beaucoup, le jugement compterait plus que les données.

# %%
renewal_names, concession_names = list(MIXTURE.renewal_readings), list(MIXTURE.concession_paths)
run_medians = mixture["frame"].assign(effective=[np.median(mixture["runs"][key]["effective"]) for key in mixture["frame"].run],
                                      contract=[np.median(mixture["runs"][key]["contract"]) for key in mixture["frame"].run])
grid_effective = run_medians.pivot(index="renouvellements", columns="concessions", values="effective").loc[renewal_names, concession_names]
grid_contract = run_medians.pivot(index="renouvellements", columns="concessions", values="contract").loc[renewal_names, concession_names]
NAMED_SCENARIOS = {"bas (cliquet)": (renewal_names[0], concession_names[-1]), "coin indexé + pourcentage de base": (renewal_names[0], concession_names[0]),
                   "haut (ancienne méthode)": (renewal_names[-1], concession_names[0])}
scenarios = pd.DataFrame({name: {"renouvellements": reading, "concessions": path, "contractuel (médiane)": grid_contract.loc[reading, path],
                                 "effectif (médiane)": grid_effective.loc[reading, path]} for name, (reading, path) in NAMED_SCENARIOS.items()}).T
print("Hausse effective (médiane) par case de la grille :")
display(grid_effective.round(CONFIG.table_digits))
display(scenarios)
marginal_effects = pd.concat({dimension: run_medians.groupby(dimension).apply(lambda group: np.average(group.effective, weights=group.poids), include_groups=False)
                              for dimension in ("renouvellements", "concessions")}).rename("effectif (médiane, moyenne pondérée sur l'autre dimension)")
display(marginal_effects.round(CONFIG.table_digits).to_frame())
prior_options = {"loi seule (concessions uniformes)": ({LAW_READING: 1}, uniform_prior(concession_names)),
                 "concessions indexées seules (renouvellements uniformes)": (uniform_prior(renewal_names), {INDEXED_PATH: 1}),
                 "règles observées seules (loi, concessions indexées)": ({LAW_READING: 1}, {INDEXED_PATH: 1})}
prior_variants = {"a priori uniforme (headline)": mixture,
                  **{name: forecast_mixture(leases, asking, EXTERNAL_TABLES, CONFIG.forecast_year, FORECAST.live_vintage, priors=priors) for name, priors in prior_options.items()}}
prior_table = pd.DataFrame({name: {"effectif (médiane)": result["effective_p50"], "contractuel (médiane)": result["contract_p50"]} for name, result in prior_variants.items()}).T
prior_table.round(CONFIG.table_digits)

# %%
renewal_effect = marginal_effects.loc["renouvellements"].max() - marginal_effects.loc["renouvellements"].min()
concession_effect_range = marginal_effects.loc["concessions"].max() - marginal_effects.loc["concessions"].min()
cell_mass = 1 / (len(renewal_names) * len(concession_names))
narrate(f"**Lecture.** La grille va de {fmt(grid_effective.min().min())} % à {fmt(grid_effective.max().max())} % de hausse effective. En moyenne sur l'autre dimension, "
        f"la trajectoire des concessions déplace le résultat de {fmt(concession_effect_range)} point et la lecture du TAL de {fmt(renewal_effect)} point : *la hausse effective de "
        f"{CONFIG.forecast_year} se joue dans deux décisions de l'entreprise*, sa politique de concessions (indexée jusqu'ici sur l'inflation des loyers de l'année précédente) et sa "
        f"lecture du changement de méthode du TAL. Les coins nommés sont des queues de distribution (chacune des {len(renewal_names) * len(concession_names)} cases pèse {fmt(cell_mass * 100)} %). "
        f"Si l'on ne retient que les règles observées (la loi, des concessions indexées), la réponse monte à {fmt(prior_table.iloc[-1, 0])} % ; selon la lecture retenue, "
        f"elle va de {fmt(prior_table.iloc[:, 0].min())} à {fmt(prior_table.iloc[:, 0].max())} % : c'est la sensibilité de la réponse au jugement.")

# %% [markdown]
# ### 11c. Intervalles : trois sources d'incertitude, mesurées séparément
# 1. **Échantillonnage** : quelles unités renouvellent, dispersion des locataires (écart-type des médianes simulées à l'intérieur d'une case).
# 2. **Structure** : les deux choix de politique incertains de la section 10d (écart-type pondéré entre les cases de la grille).
# 3. **Erreur de modèle** : l'erreur quadratique moyenne du backtest du **même modèle, au même millésime**, sur les années exigées. Elle couvre ce que le modèle ne capte pas d'une année à l'autre (relocations autour de l'IPC, part de renouvellements…) ; elle est *optimiste*, car la structure a été choisie sur ces années.
#
# La **distribution prédictive** ajoute l'erreur de modèle (loi normale) aux tirages du modèle ; le headline reste la médiane du modèle structurel ; les percentiles prédictifs `ci_low`/`ci_high` forment la bande. **Risque de régime, à part :** hors du régime récent, le modèle sous-prédisait nettement (erreurs toutes du même signe), ce que la bande symétrique ne montre pas.

# %%
structural_live_rows = backtest_table[(backtest_table.model == "structural") & (backtest_table.vintage == FORECAST.live_vintage)].set_index("year")
model_sd = {kind: float(np.sqrt((structural_live_rows.loc[list(CONFIG.backtest_years), f"error_{kind}"] ** 2).mean())) for kind in ("effective", "contract")}
model_bias = {kind: float(structural_live_rows.loc[list(CONFIG.backtest_years), f"error_{kind}"].mean()) for kind in ("effective", "contract")}
bias_sensitivity = pd.DataFrame({kind: {"médiane structurelle (%)": mixture[f"{kind}_p50"],
    "biais moyen prévu − réalisé (points)": model_bias[kind],
    "sensibilité : médiane moins biais (%)": mixture[f"{kind}_p50"] - model_bias[kind]}
    for kind in ("effective", "contract")}).T
display(bias_sensitivity.round(CONFIG.table_digits))
narrate(f"**Biais récent.** Le modèle surestime l’effectif en moyenne de {fmt(model_bias['effective'], CONFIG.table_digits)} point ; "
        f"soustraire ce biais mettrait la médiane à {fmt(bias_sensitivity.loc['effective', 'sensibilité : médiane moins biais (%)'], CONFIG.table_digits)} %. "
        "C’est une sensibilité, pas une correction validée : les mêmes années ont servi à repérer la structure. "
        "Le headline reste structurel et la bande utilise l’erreur quadratique totale, biais inclus.")
noise_rng = np.random.default_rng(CONFIG.random_seed)
predictive = {kind: pooled[kind] + noise_rng.normal(0, model_sd[kind], len(pooled[kind])) for kind in ("effective", "contract")}


def run_spread(kind):
    """Écarts-types d'échantillonnage (dans une case) et de structure (entre cases), pondérés par les a priori."""
    weights = mixture["frame"].poids.to_numpy()
    means = np.array([mixture["runs"][key][kind].mean() for key in mixture["frame"].run])
    within = np.array([mixture["runs"][key][kind].std() for key in mixture["frame"].run])
    center = np.average(means, weights=weights)
    return float(np.sqrt(np.average(within ** 2, weights=weights))), float(np.sqrt(np.average((means - center) ** 2, weights=weights)))


uncertainty_rows = {}
for kind, label in (("effective", "effectif"), ("contract", "contractuel")):
    sampling_sd, structural_sd = run_spread(kind)
    uncertainty_rows[label] = {
        "écart-type : échantillonnage": sampling_sd, "écart-type : structure (grille)": structural_sd, "écart-type : erreur de modèle (backtest)": model_sd[kind],
        f"p{CONFIG.ci_low:.0f} (modèle seul)": weighted_percentile(pooled[kind], pooled_weights, CONFIG.ci_low),
        "médiane": weighted_percentile(pooled[kind], pooled_weights, MEDIAN_PERCENTILE),
        f"p{CONFIG.ci_high:.0f} (modèle seul)": weighted_percentile(pooled[kind], pooled_weights, CONFIG.ci_high),
        f"p{CONFIG.ci_low:.0f} (prédictive)": weighted_percentile(predictive[kind], pooled_weights, CONFIG.ci_low),
        f"p{CONFIG.ci_high:.0f} (prédictive)": weighted_percentile(predictive[kind], pooled_weights, CONFIG.ci_high),
        "moyenne pondérée par le loyer": weighted_percentile(pooled[f"{kind}_mean"], pooled_weights, MEDIAN_PERCENTILE)}
uncertainty = pd.DataFrame(uncertainty_rows).T
mc_headlines = [forecast_mixture(leases, asking, EXTERNAL_TABLES, CONFIG.forecast_year, FORECAST.live_vintage, seed=CONFIG.random_seed + offset)["effective_p50"]
                for offset in range(1, FORECAST.mc_precision_seeds + 1)]
mc_precision = float(np.std(mc_headlines, ddof=1))
print(f"Précision Monte-Carlo du headline : {FORECAST.mc_precision_seeds} graines indépendantes → {np.round(mc_headlines, CONFIG.table_digits)} (écart-type {mc_precision:.3f} point)")
display(uncertainty.round(CONFIG.table_digits))
early_years = [year for year in CONFIG.extended_backtest_years if year not in CONFIG.backtest_years]
regime_errors = structural_live_rows.loc[early_years, ["error_contract", "error_effective"]]
print("Risque de régime (erreurs prévu − réalisé hors du régime récent) :")
display(regime_errors.round(CONFIG.table_digits))

# %%
effective_p50, contract_p50 = mixture["effective_p50"], mixture["contract_p50"]
concession_effect_mean = float(np.average(pooled["concession_effect"], weights=pooled_weights))
attribution = pd.Series({"hausse contractuelle (médiane des paires)": contract_p50, "+ effet moyen des concessions sur les paires": concession_effect_mean,
                         "+ effet de forme de la médiane": effective_p50 - contract_p50 - concession_effect_mean, "= hausse effective (médiane des paires)": effective_p50},
                        name="points de %")
display(attribution.round(CONFIG.table_digits).to_frame())
low_label, high_label = f"p{CONFIG.ci_low:.0f} (prédictive)", f"p{CONFIG.ci_high:.0f} (prédictive)"
narrate(f"**Lecture.** La distribution prédictive de la hausse effective va de **{fmt(uncertainty.loc['effectif', low_label])} % à {fmt(uncertainty.loc['effectif', high_label])} %** "
        f"(bande {CONFIG.ci_low:.0f}-{CONFIG.ci_high:.0f} %), autour d'une médiane de {fmt(effective_p50)} %. La plus grande source d'incertitude est "
        f"{'la structure, c’est-à-dire les deux inconnues de 2026' if uncertainty.loc['effectif', 'écart-type : structure (grille)'] > model_sd['effective'] else 'l’erreur de modèle'} "
        f"(écarts-types : structure {fmt(uncertainty.loc['effectif', 'écart-type : structure (grille)'], CONFIG.table_digits)}, erreur de modèle {fmt(model_sd['effective'], CONFIG.table_digits)}, "
        f"échantillonnage {fmt(uncertainty.loc['effectif', 'écart-type : échantillonnage'], CONFIG.table_digits)}). Le nombre de tirages suffit : relancé avec "
        f"{FORECAST.mc_precision_seeds} autres graines, le headline varie de {fmt(mc_precision, CONFIG.table_digits)} point (écart-type). La mesure historique est précise ; "
        "c'est la prévision qui est incertaine.\n\n"
        f"**D'où vient l'écart entre contractuel et effectif ?** La hausse contractuelle médiane ({fmt(contract_p50)} %) plus l'effet moyen des concessions sur les paires "
        f"({fmt(concession_effect_mean, signed=True)} point) plus un effet de forme de la médiane ({fmt(attribution.iloc[2], signed=True)} point) donnent exactement la hausse effective "
        f"médiane ({fmt(effective_p50)} %). L'effet de forme n'est pas un revenu : il vient de ce que la médiane des paires change de groupe (renouvellements, relocations) quand la "
        f"dispersion des concessions s'ajoute ; c'est pourquoi la **moyenne pondérée par le loyer** ({fmt(uncertainty.loc['effectif', 'moyenne pondérée par le loyer'])} %) est publiée à côté.\n\n"
        f"**Risque de régime.** Hors du régime récent ({', '.join(map(str, early_years))}), le modèle se trompait de {', '.join(fmt(value, signed=True) for value in regime_errors.error_effective)} "
        "point(s) sur l'effectif, toujours vers le bas : si la société revenait à des renouvellements au-dessus du TAL, la hausse serait plus forte que la bande ne le dit.")

# %%
fig, ax = plt.subplots(figsize=STYLE.figsize_wide)
history_years = range(CONFIG.first_year, CONFIG.last_observed_year + 1)
ax.plot(history_years, same_unit.loc[history_years, ("contractuel", "median")], "o-", color=PALETTE["contract"], label="contractuel réalisé")
ax.plot(history_years, same_unit.loc[history_years, ("effectif", "median")], "o-", color=PALETTE["effective"], label="effectif réalisé")
for kind, label, offset in (("contract", "contractuel", -STYLE.marker_jitter_years), ("effective", "effectif", STYLE.marker_jitter_years)):
    row = uncertainty.loc[label]
    ax.errorbar(CONFIG.forecast_year + offset, row["médiane"], yerr=[[row["médiane"] - row[low_label]], [row[high_label] - row["médiane"]]],
                fmt="D", color=PALETTE[kind], capsize=STYLE.error_capsize, label=f"{CONFIG.forecast_year} prévu ({label}, bande prédictive)")
for name, marker in zip(NAMED_SCENARIOS, ("v", "s", "^")):
    ax.plot(CONFIG.forecast_year, scenarios.loc[name, "effectif (médiane)"], marker, color="black", fillstyle="none", label=f"scénario {name} (effectif)")
ax.set(title="Hausse à unité constante, médiane des paires (%)", xlabel="année de début de bail")
ax.legend(fontsize=STYLE.small_fontsize, loc="upper left")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 11d. The Met sous les règles de l'Ontario : la valeur de l'exemption
# Régime central : *exempté* (section 6d). Régime contrefactuel : les renouvellements de The Met plafonnés à la ligne directrice de l'année prévue. Même case de la grille (règles observées : la loi, des concessions indexées) et mêmes tirages aléatoires dans les deux cas, pour que la différence ne vienne que de la règle.

# %%
guideline_2026 = float(ONTARIO_GUIDELINE[CONFIG.forecast_year])
exempt = run_scenario(LAW_READING, mixture["paths"][INDEXED_PATH])
capped = run_scenario(LAW_READING, mixture["paths"][INDEXED_PATH], ontario_regime="capped", ontario_guideline=guideline_2026)
met_columns = (exempt["cohort"].province == "Ontario").to_numpy()
met_rents = exempt["weights"][met_columns]
value_of_exemption_pct = float(((exempt["contract"][:, met_columns] - capped["contract"][:, met_columns]) @ met_rents).mean() / met_rents.sum())
value_of_exemption_dollars = value_of_exemption_pct / 100 * met_rents.sum() * CONFIG.months_per_year
ontario_regimes = pd.DataFrame({
    "The Met : hausse contractuelle (moyenne pondérée, %)": [(simulation["contract"][:, met_columns] @ met_rents).mean() / met_rents.sum() for simulation in (exempt, capped)],
    "portefeuille : contractuel (médiane, %)": [np.median(simulation["contract"], axis=1).mean() for simulation in (exempt, capped)],
    "portefeuille : effectif (médiane, %)": [np.median(simulation["effective"], axis=1).mean() for simulation in (exempt, capped)]},
    index=["exempté (central)", f"plafonné à {guideline_2026} % (contrefactuel)"])
display(ontario_regimes.round(CONFIG.table_digits))
narrate(f"**Lecture.** {int(met_columns.sum())} unités de The Met échoient en {CONFIG.forecast_year} ({fmt(met_columns.mean() * 100, 0)} % de la cohorte). "
        f"L'exemption vaut **{fmt(value_of_exemption_pct, CONFIG.table_digits, signed=True)} point de hausse contractuelle** sur le loyer de The Met, soit environ "
        f"{fmt(value_of_exemption_dollars, 0)} $ de loyer contractuel annuel. L'effet sur la médiane du portefeuille est de "
        f"{fmt(ontario_regimes.iloc[0, 2] - ontario_regimes.iloc[1, 2], CONFIG.table_digits, signed=True)} point (effectif). Nous gardons les deux régimes visibles. "
        "**The Met, exempté de la ligne directrice, reçoit la politique de renouvellement de la société ; la ligne directrice de l'Ontario n'apparaît que comme contrefactuel "
        "et n'est jamais appliquée aux cinq immeubles du Québec.**")

# %% [markdown]
# ### 11e. Par propriété et par nombre de chambres (bonus)
# Mêmes simulations que le headline, regroupées : pour chaque segment, la médiane des paires simulées de ses unités, sur toutes les cases de la grille pondérées par leurs a priori. **Règle de petites cellules** : une propriété × chambres avec moins de `min_cell_size` unités n'est pas publiée seule ; elle affiche la valeur de sa propriété, signalée. Les sorties n'ont aucun identifiant d'unité (**aucune donnée CRM**).

# %%
property_names = leases.groupby("prop_code").building.first()
segment_ids = list(next(iter(mixture["runs"].values()))["segments"])


def pooled_segment(segment_id, statistic):
    values = [mixture["runs"][row.run]["segments"][segment_id][statistic] for row in mixture["frame"].itertuples()]
    weights = [np.full(len(draws), row.poids / len(draws)) for draws, row in zip(values, mixture["frame"].itertuples())]
    return np.concatenate(values), np.concatenate(weights)


def segment_row(segment_id, last_year_pairs_segment):
    effective_draws, weights = pooled_segment(segment_id, "effective")
    contract_draws, _ = pooled_segment(segment_id, "contract")
    return {"unités 2026": mixture["runs"][mixture["frame"].run.iloc[0]]["segments"][segment_id]["units"],
            "contractuel": weighted_percentile(contract_draws, weights, MEDIAN_PERCENTILE), "effectif": weighted_percentile(effective_draws, weights, MEDIAN_PERCENTILE),
            f"effectif bas ({CONFIG.ci_low:.0f} %)": weighted_percentile(effective_draws, weights, CONFIG.ci_low),
            f"effectif haut ({CONFIG.ci_high:.0f} %)": weighted_percentile(effective_draws, weights, CONFIG.ci_high),
            f"{CONFIG.last_observed_year} réalisé (effectif)": last_year_pairs_segment.growth_effective_pct.median(), "estimation reportée à la propriété": False}


property_rows = {}
for keys, label in [segment_id for segment_id in segment_ids if segment_id[0] == ("prop_code",)]:
    code = label[0]
    property_rows[code] = {"propriété": f"{property_names[code]} ({code})", "chambres": "toutes",
                           **segment_row((keys, label), last_year_pairs[last_year_pairs.prop_code == code])}
segment_rows = list(property_rows.values())
for keys, label in [segment_id for segment_id in segment_ids if segment_id[0] == ("prop_code", "beds")]:
    code, beds = label
    row = segment_row((keys, label), last_year_pairs[(last_year_pairs.prop_code == code) & (last_year_pairs.beds == beds)])
    if row["unités 2026"] < CONFIG.min_cell_size:
        row = {**property_rows[code], "unités 2026": row["unités 2026"], "estimation reportée à la propriété": True}
    segment_rows.append({**row, "propriété": f"{property_names[code]} ({code})", "chambres": int(beds)})
aggregates_2026 = pd.DataFrame(segment_rows).sort_values(["propriété", "chambres"], key=lambda column: column.astype(str)).reset_index(drop=True)
aggregates_2026.round(CONFIG.table_digits)

# %%
by_property = aggregates_2026[aggregates_2026.chambres == "toutes"].set_index("propriété")
narrate(f"**Lecture.** La hausse effective médiane va de {fmt(by_property.effectif.min())} % ({by_property.effectif.idxmin()}) à {fmt(by_property.effectif.max())} % "
        f"({by_property.effectif.idxmax()}). Les écarts entre propriétés viennent de la **composition connue** de chaque cohorte (concessions des baux qui échoient, durée, "
        "part de renouvellements) et de la distribution des concessions propre à chaque propriété ; ils restent petits devant l'incertitude globale. Les cellules à peu d'unités "
        "sont rapportées au niveau de la propriété plutôt que publiées seules, parce qu'une médiane sur quelques unités est du bruit.")

# %% [markdown]
# **Les écarts entre segments persistent-ils d'une année à l'autre ?** Si une propriété ou un nombre de chambres croissait durablement plus vite que le portefeuille, la prévision par segment devrait en tenir compte. Test : l'écart (médiane effective du segment − médiane du portefeuille) de l'année `t−1` prédit-il celui de `t` ? On compare à l'hypothèse « aucun écart » sur les segments d'au moins `min_cell_size` paires.

# %%
segment_year = pairs.groupby(["prop_code", "beds", "lease_year"]).growth_effective_pct.agg(median="median", pairs="size").reset_index()
segment_year = segment_year[segment_year.pairs >= CONFIG.min_cell_size]
segment_year["deviation"] = segment_year["median"] - segment_year.lease_year.map(pairs.groupby("lease_year").growth_effective_pct.median())
previous_deviation = segment_year.assign(lease_year=segment_year.lease_year + 1)[["prop_code", "beds", "lease_year", "deviation"]].rename(columns={"deviation": "deviation_previous_year"})
persistence_test = segment_year.merge(previous_deviation, on=["prop_code", "beds", "lease_year"])
persistence_test = persistence_test[persistence_test.lease_year >= CONFIG.backtest_years[0]]
persistence_labels = {"tested": "segments testés (segment × année)", "correlation": "corrélation écart(t) × écart(t−1)",
                      "no_deviation": "erreur si l'on prédit « aucun écart » (points)", "carry_over": "erreur si l'on reconduit l'écart de l'an dernier (points)"}
persistence_summary = pd.Series({
    persistence_labels["tested"]: len(persistence_test),
    persistence_labels["correlation"]: persistence_test.deviation.corr(persistence_test.deviation_previous_year),
    persistence_labels["no_deviation"]: persistence_test.deviation.abs().mean(),
    persistence_labels["carry_over"]: (persistence_test.deviation - persistence_test.deviation_previous_year).abs().mean()})
display(persistence_summary.round(CONFIG.table_digits).to_frame("valeur"))
persistence = {key: persistence_summary[label] for key, label in persistence_labels.items()}
reconduction_worse = persistence["carry_over"] > persistence["no_deviation"]
narrate(f"**Lecture.** Sur {int(persistence['tested'])} segments × années, la corrélation entre l'écart d'une année et celui de la suivante vaut "
        f"{fmt(persistence['correlation'], CONFIG.table_digits)}, et reconduire l'écart de l'an dernier donne une erreur de {fmt(persistence['carry_over'], CONFIG.table_digits)} point "
        f"contre {fmt(persistence['no_deviation'], CONFIG.table_digits)} si l'on n'en prédit aucun : "
        + ("**reconduire est pire**. La prévision par segment ne diffère donc du portefeuille que par ce qui est connu de chaque cohorte, ce que fait notre simulation : "
           "c'est une conclusion, pas une paresse de modélisation." if reconduction_worse else
           "les écarts persistent en partie, une piste d'amélioration pour une prévision par segment."))

# %%
aggregates_2026.to_csv(CONFIG.output_dir / "aggregates_2026.csv", index=False)
forbidden_columns = {"unit_code", "unit_id", "lease_id", "sign_date", "lease_start", "rent_contract", "rent_effective"}
assert not forbidden_columns & set(aggregates_2026.columns), "aucun identifiant d'unité ni loyer de bail dans les sorties"
assert (aggregates_2026[~aggregates_2026["estimation reportée à la propriété"]]["unités 2026"] >= CONFIG.min_cell_size).all(), "règle de petites cellules"
if plotly_go is not None:
    property_view = aggregates_2026[aggregates_2026.chambres == "toutes"]
    high_column, low_column = f"effectif haut ({CONFIG.ci_high:.0f} %)", f"effectif bas ({CONFIG.ci_low:.0f} %)"
    dashboard = make_subplots(rows=1, cols=2, subplot_titles=(f"Hausse {CONFIG.forecast_year} par propriété (médiane, bande {CONFIG.ci_low:.0f}-{CONFIG.ci_high:.0f} %)",
                                                              "Effectif par propriété × chambres"))
    dashboard.add_trace(plotly_go.Bar(x=property_view["propriété"], y=property_view["effectif"], name="effectif",
                                      error_y=dict(type="data", symmetric=False, array=property_view[high_column] - property_view["effectif"],
                                                   arrayminus=property_view["effectif"] - property_view[low_column]), marker_color=PALETTE["effective"]), row=1, col=1)
    dashboard.add_trace(plotly_go.Bar(x=property_view["propriété"], y=property_view["contractuel"], name="contractuel", marker_color=PALETTE["contract"]), row=1, col=1)
    detail_view = aggregates_2026[aggregates_2026.chambres != "toutes"]
    dashboard.add_trace(plotly_go.Heatmap(x=detail_view["propriété"], y=detail_view.chambres.astype(str), z=detail_view["effectif"], colorscale="RdYlGn",
                                          colorbar=dict(title="%")), row=1, col=2)
    dashboard.update_layout(title=f"Collection Équinoxe : hausse de loyer {CONFIG.forecast_year} (agrégats seulement, aucune donnée CRM)", barmode="group")
    dashboard.write_html(CONFIG.output_dir / "dashboard_2026.html", include_plotlyjs="cdn")
    print("Tableau de bord écrit : outputs/dashboard_2026.html (agrégats seulement)")
else:
    print("plotly absent : le tableau de bord HTML est facultatif ; les agrégats sont dans outputs/aggregates_2026.csv")

# %% [markdown]
# ### 11f. Réponse finale
# Le headline est la **médiane du modèle structurel** de la hausse effective ; la bande est celle de 11c. Le fichier `outputs/final_answer.json` reprend ces valeurs (agrégats seulement).

# %%
headline = effective_p50
final_answer = {
    "definition": f"Croissance à unité constante du loyer effectif (net des concessions), médiane des paires, baux débutant en {CONFIG.forecast_year} (renouvellements + relocations)",
    "headline_effective_pct": round(headline, CONFIG.table_digits),
    "headline_band_pct": [round(uncertainty.loc["effectif", low_label], CONFIG.table_digits), round(uncertainty.loc["effectif", high_label], CONFIG.table_digits)],
    "headline_method": "modèle structurel (TAL daté par l'avis, IPC loyers de l'année précédente, concessions à deux marges) ; grille des inconnues de 2026 pondérée uniformément ; "
                       "bande = grille + échantillonnage + erreur de modèle du backtest",
    "effective_weighted_mean_pct": round(uncertainty.loc["effectif", "moyenne pondérée par le loyer"], CONFIG.table_digits),
    "contract_pct": round(contract_p50, CONFIG.table_digits),
    "contract_band_pct": [round(uncertainty.loc["contractuel", low_label], CONFIG.table_digits), round(uncertainty.loc["contractuel", high_label], CONFIG.table_digits)],
    "renewal_anchor_law_pct": round(float(law_anchor.mean()), CONFIG.table_digits),
    "attribution_points": {name: round(value, CONFIG.table_digits) for name, value in attribution.items()},
    "scenarios_effective_pct": {name: round(float(value), CONFIG.table_digits) for name, value in scenarios["effectif (médiane)"].items()},
    "prior_sensitivity_effective_pct": {name: round(float(value), CONFIG.table_digits) for name, value in prior_table["effectif (médiane)"].items()},
    "backtest_bias_points": {kind: round(value, CONFIG.table_digits) for kind, value in model_bias.items()},
    "bias_adjusted_sensitivity_pct": {kind: round(float(value), CONFIG.table_digits) for kind, value in bias_sensitivity["sensibilité : médiane moins biais (%)"].items()},
    "robustness_effective_range_pct": [round(float(robustness.iloc[:, 0].min()), CONFIG.table_digits), round(float(robustness.iloc[:, 0].max()), CONFIG.table_digits)],
    "mc_precision_sd_points": round(mc_precision, CONFIG.fine_digits),
    "backtest_mae_points": {kind: round(float(structural_live_rows.loc[list(CONFIG.backtest_years), f"error_{kind}"].abs().mean()), CONFIG.table_digits) for kind in ("effective", "contract")},
    "presentation_diagnostics": {"mix_year": int(naive_peak_year), "naive_growth_pct": round(float(naive.yoy_pct[naive_peak_year]), CONFIG.table_digits),
        "fixed_panel_growth_pct": round(float(panel_yoy[naive_peak_year]), CONFIG.table_digits), "fisher_growth_pct": round(float(fisher[naive_peak_year]), CONFIG.table_digits),
        "reconstruction_share_pct": round(float(close.mean() * 100), CONFIG.table_digits), "reconstruction_tolerance_dollars": CONFIG.exact_tolerance_dollars,
        "concession_penetration_pct": round(float(profile_by_year.penetration_pct[CONFIG.last_observed_year]), CONFIG.table_digits)},
    "information_vintage": FORECAST.live_vintage, "data_as_of": CONFIG.data_as_of, "cohort_units": int(mixture["n_units"])}
(CONFIG.output_dir / "final_answer.json").write_text(json.dumps(final_answer, ensure_ascii=False, indent=2), encoding="utf-8")
# Tables agrégées utilisées par les graphiques du jury et la vérification de livraison.
evidence = {
    "backtest": backtest_table.drop(columns=["centers", "actual_components"]).to_dict(orient="records"),
    "baselines": baseline_table.to_dict(orient="records"),
    "historical_growth": [{"year": int(year), "contract": float(same_unit.loc[year, ("contractuel", "median")]),
        "effective": float(same_unit.loc[year, ("effectif", "median")])} for year in range(CONFIG.first_year, CONFIG.last_observed_year + 1)],
    "pair_counts": {str(year): int(count) for year, count in pairs.groupby("lease_year").size().items()},
    "uncertainty": uncertainty.reset_index(names="kind").to_dict(orient="records"),
    "robustness": robustness.reset_index(names="setting").to_dict(orient="records"),
    "policy_sensitivity": prior_table.reset_index(names="policy").to_dict(orient="records"),
    "sources": sorted(set(external.source_url)),
    "annualisation": "Exact elapsed days between lease starts / days_per_year, matching build_pairs",
    "data_as_of": CONFIG.data_as_of,
    "backtest_years": list(CONFIG.backtest_years), "live_vintage": FORECAST.live_vintage}
(CONFIG.output_dir / "evidence.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

print(f"HAUSSE DE LOYER {CONFIG.forecast_year} (définition principale) : {headline:.1f} %  [bande {CONFIG.ci_low:.0f}-{CONFIG.ci_high:.0f} % : "
      f"{final_answer['headline_band_pct'][0]:.1f} à {final_answer['headline_band_pct'][1]:.1f} %]")
print(f"   {final_answer['definition']}")
print(f"HAUSSE CONTRACTUELLE {CONFIG.forecast_year} (même unité, loyer au bail) : {final_answer['contract_pct']:.1f} %  "
      f"[{final_answer['contract_band_pct'][0]:.1f} à {final_answer['contract_band_pct'][1]:.1f} %]")

# %%
narrate(f"### Réponse : **{fmt(headline)} %** de hausse effective en {CONFIG.forecast_year} (bande {CONFIG.ci_low:.0f}-{CONFIG.ci_high:.0f} % : "
        f"{fmt(final_answer['headline_band_pct'][0])} à {fmt(final_answer['headline_band_pct'][1])} %)\n\n"
        f"1. **Contractuel : {fmt(contract_p50)} %** ({fmt(final_answer['contract_band_pct'][0])} à {fmt(final_answer['contract_band_pct'][1])} %). Les renouvellements suivent "
        f"une politique commune ancrée sur le calendrier du TAL (proxy pour The Met, pas droit québécois) (en moyenne {fmt(law_anchor.mean(), CONFIG.table_digits)} % : ni le pourcentage de base de "
        f"{fmt(TAL_NEW_FORMAT[CONFIG.forecast_year])} % pour tous, ni l'ancienne méthode) ; les relocations suivent l'IPC loyers du Québec de l'année précédente "
        f"({fmt(mixture['lagged_cpi'], CONFIG.table_digits)} %).\n"
        f"2. **Effectif : {fmt(headline)} %**, au-dessus du contractuel parce que les concessions des nouveaux baux devraient se situer sous ou au niveau de celles des baux qui échoient "
        f"(drag {CONFIG.last_observed_year} : {fmt(mixture['last_drag'])} % ; indexé sur l'IPC : {fmt(mixture['lagged_cpi'])} %). Moyenne pondérée par le loyer : "
        f"{fmt(final_answer['effective_weighted_mean_pct'])} %.\n"
        f"3. **Ce qui est jugement :** deux choix de politique incertains (suivi de la loi par la société, comportement des concessions quand l'IPC baisse), pondérées uniformément "
        f"faute de preuve suffisante pour les départager ; selon la lecture retenue, la réponse va de {fmt(prior_table['effectif (médiane)'].min())} à {fmt(prior_table['effectif (médiane)'].max())} % "
        f"({fmt(prior_table.iloc[-1, 0])} % si l'on ne retient que les règles observées).\n"
        f"4. **Ce qui est mesuré :** le même modèle, rejoué sur {CONFIG.backtest_years[0]}-{CONFIG.backtest_years[-1]}, se trompe en moyenne de "
        f"{fmt(final_answer['backtest_mae_points']['effective'], CONFIG.table_digits)} point sur l'effectif et {fmt(final_answer['backtest_mae_points']['contract'], CONFIG.table_digits)} "
        "point sur le contractuel ; en cas de changement de régime, l'erreur historique est plus grande et à sens unique (11c).")

# %%
assert final_answer["headline_band_pct"][0] <= final_answer["headline_effective_pct"] <= final_answer["headline_band_pct"][1]
scenario_values = [scenarios.loc[name, "effectif (médiane)"] for name in NAMED_SCENARIOS]
assert scenario_values == sorted(scenario_values), "scénarios ordonnés du bas vers le haut"
observed_rules_cell = grid_effective.loc[LAW_READING, INDEXED_PATH]
assert final_answer["headline_band_pct"][0] <= observed_rules_cell <= final_answer["headline_band_pct"][1], "la bande couvre la case des règles observées"
assert predictive["effective"].min() <= min(scenario_values) and max(scenario_values) <= predictive["effective"].max(), "les coins nommés sont dans le support prédictif"
assert np.isclose(estimate_2026(leases, asking), headline), "estimate_2026() doit reproduire exactement la réponse (mêmes nombres aléatoires)"
print("Contrôles de la section 11 : OK")

# %% [markdown]
# ### 11g. Aperçu 2027 : le prochain budget est déjà lisible dans les données publiques
# Les règles qui pilotent le modèle sont publiques **avant** que l'année commence : le pourcentage de base du TAL (moyenne des trois variations annuelles de l'IPC d'ensemble du Québec) et l'IPC loyers de l'année écoulée (qui fixe les relocations et les concessions). Sans changer le modèle, on peut donc lire 2027 dès aujourd'hui, avec les mois déjà publiés.

# %%
next_year = CONFIG.forecast_year + 1
all_items_ytd = external_series("cpi_all_items_ytd_yoy", "Quebec")
all_items_with_current = pd.concat([all_items, all_items_ytd.loc[[CONFIG.forecast_year]]])
tal_next_year = max(float(all_items_with_current.loc[next_year - FORECAST.tal_new_method_window_years:next_year - 1].mean()), 0.0)
rent_cpi_current = cpi_driver(EXTERNAL_TABLES, CONFIG.forecast_year, live_cutoff)
current_year_cpi = cpi_monthly[(cpi_monthly.geo == "Quebec") & (cpi_monthly.month.dt.year == CONFIG.forecast_year)].set_index("month").yoy_pct
preview_next_year = pd.DataFrame({
    f"{CONFIG.forecast_year} (prévu)": [TAL_NEW_FORMAT[CONFIG.forecast_year], mixture["lagged_cpi"], mixture["centers"]["drag"]],
    f"{next_year} (aperçu)": [tal_next_year, rent_cpi_current, rent_cpi_current]},
    index=["renouvellements : pourcentage de base du TAL (%)", "relocations : IPC loyers QC de l'année précédente (%)", "concessions : drag moyen de la grille (%)"])
print(f"IPC publié jusqu'en {current_year_cpi.index.max():%Y-%m} ; variation annuelle de l'IPC loyers QC, mois par mois :", current_year_cpi.round(CONFIG.prose_digits).to_dict())
preview_next_year.round(CONFIG.table_digits)

# %%
narrate(f"**Lecture.** Sous les règles observées, le pourcentage de base du TAL passerait de {fmt(TAL_NEW_FORMAT[CONFIG.forecast_year])} % à **{fmt(tal_next_year)} %** en {next_year}, "
        f"et les relocations comme les concessions (liées à l'IPC loyers de {CONFIG.forecast_year}, qui décélère de {fmt(current_year_cpi.iloc[0])} % en janvier à "
        f"{fmt(current_year_cpi.iloc[-1])} % au dernier mois publié) à **{fmt(rent_cpi_current)} %** contre {fmt(mixture['lagged_cpi'])} % pour {CONFIG.forecast_year}. "
        "C'est un **aperçu**, pas une prévision validée : il suppose que les règles tiennent encore et que les mois restants ressemblent aux mois déjà publiés.")

# %%
assert 0 < tal_next_year < TAL_NEW_FORMAT[CONFIG.forecast_year] + 1, "le pourcentage de base de l'année suivante doit rester plausible"
print("Contrôles de la section 11g : OK")
