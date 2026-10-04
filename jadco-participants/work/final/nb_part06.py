# %% [markdown]
# ## 7. Choisir la définition : tableau comparatif
#
# La section 1 a annoncé le choix ; voici **la preuve**. Les cinq définitions sont calculées sur les mêmes données et notées selon des critères explicites, au lieu de plaider dans l'abstrait.
#
# | Critère | Question | Mesure |
# |---|---|---|
# | **Insensibilité au mix** | Le nombre bouge-t-il quand le portefeuille change ? | écart moyen sur les années du backtest entre la version « tous immeubles » et la version « panel fixe » |
# | **Bruit** | Quelle précision ? | largeur de l'intervalle (`ci_low`–`ci_high`) d'un bootstrap **par unité** (les paires d'une même unité sont corrélées) la dernière année |
# | **Prévisibilité** | Peut-on le prévoir ? | erreur absolue moyenne de deux références simples (reconduction de l'an dernier, moyenne des `recent_years_for_mean` dernières années) |
# | **Lecture réglementaire** | Comparable au TAL ou à la ligne directrice ? | oui seulement pour le loyer contractuel des renouvellements |
# | **Usage de décision** | Quelle ligne budgétaire ? | loyer encaissé (budget) ou loyer au bail (lettres de renouvellement) |
# | **Soutien des données** | Combien de paires ? | nombre de paires la dernière année |
#
# **Cette étape ajoute une colonne** à `pairs` : `unit_label` (`prop_code|unit_code`), l'identifiant de grappe du bootstrap.

# %%
pairs["unit_label"] = pairs.prop_code + "|" + pairs.unit_code
panel_pairs = pairs[pairs.prop_code.isin(panel_codes)]
DEFINITIONS = {
    "1. même unité, contractuel": lambda frame: frame.groupby("lease_year").growth_contract_pct.median(),
    "2. même unité, effectif (RETENU)": lambda frame: frame.groupby("lease_year").growth_effective_pct.median(),
    "3. renouvellements seulement (contractuel)": lambda frame: frame[frame.is_renewal == 1].groupby("lease_year").growth_contract_pct.median(),
    "4. relocations seulement (contractuel)": lambda frame: frame[frame.is_renewal == 0].groupby("lease_year").growth_contract_pct.median(),
}
NAIVE_DEFINITION = "5. médiane du portefeuille (contre-exemple)"
definition_series_all = {name: function(pairs) for name, function in DEFINITIONS.items()}
definition_series_panel = {name: function(panel_pairs) for name, function in DEFINITIONS.items()}
definition_series_all[NAIVE_DEFINITION] = naive.yoy_pct
definition_series_panel[NAIVE_DEFINITION] = panel_yoy
definition_table = pd.DataFrame(definition_series_all).loc[CONFIG.extended_backtest_years[0]:CONFIG.last_observed_year]
definition_table.round(CONFIG.table_digits)

# %%
def cluster_bootstrap_median(values, unit_labels, n_draws, rng):
    """Bootstrap de la médiane en rééchantillonnant les UNITÉS (pas les paires) : médiane pondérée par tirage multinomial."""
    labels, inverse = np.unique(unit_labels, return_inverse=True)
    order = np.argsort(values)
    sorted_values, sorted_inverse = values[order], inverse[order]
    counts = rng.multinomial(len(labels), np.full(len(labels), 1 / len(labels)), size=n_draws)
    cumulative = np.cumsum(counts[:, sorted_inverse], axis=1)
    half = cumulative[:, -1:] / 2
    return sorted_values[(cumulative < half).sum(axis=1)]


def interval_width(draws):
    low, high = np.percentile(draws, [CONFIG.ci_low, CONFIG.ci_high])
    return high - low


bootstrap_rng = np.random.default_rng(CONFIG.random_seed)
last_year_pairs = pairs[pairs.lease_year == CONFIG.last_observed_year]
last_year_subsets = {
    "1. même unité, contractuel": (last_year_pairs, "growth_contract_pct"),
    "2. même unité, effectif (RETENU)": (last_year_pairs, "growth_effective_pct"),
    "3. renouvellements seulement (contractuel)": (last_year_pairs[last_year_pairs.is_renewal == 1], "growth_contract_pct"),
    "4. relocations seulement (contractuel)": (last_year_pairs[last_year_pairs.is_renewal == 0], "growth_contract_pct")}
bootstrap_width = {name: interval_width(cluster_bootstrap_median(frame[column].to_numpy(), frame.unit_label.to_numpy(), CONFIG.n_bootstrap, bootstrap_rng))
                   for name, (frame, column) in last_year_subsets.items()}
# le portefeuille naïf se rééchantillonne par bail, en recalculant la variation de médiane entre les deux dernières années
previous_rents = leases[leases.lease_year == CONFIG.last_observed_year - 1].rent_contract.to_numpy()
current_rents = leases[leases.lease_year == CONFIG.last_observed_year].rent_contract.to_numpy()
naive_draws = [(np.median(bootstrap_rng.choice(current_rents, len(current_rents))) / np.median(bootstrap_rng.choice(previous_rents, len(previous_rents))) - 1) * 100
               for _ in range(CONFIG.n_bootstrap)]
bootstrap_width[NAIVE_DEFINITION] = interval_width(naive_draws)


def persistence_and_recent_mean_mae(series, years):
    errors_persistence = [series[year] - series[year - 1] for year in years]
    errors_recent_mean = [series[year] - series.loc[:year - 1].tail(CONFIG.recent_years_for_mean).mean() for year in years]
    return np.mean(np.abs(errors_persistence)), np.mean(np.abs(errors_recent_mean))


interval_label = f"intervalle {CONFIG.ci_low:.0f}-{CONFIG.ci_high:.0f} % en {CONFIG.last_observed_year} (points)"
recent_mean_label = f"erreur « moyenne des {CONFIG.recent_years_for_mean} dernières années »"
scorecard = pd.DataFrame(index=list(definition_series_all))
scorecard["écart tous immeubles vs panel fixe (points)"] = [
    float(np.mean([abs(definition_series_all[name][year] - definition_series_panel[name][year]) for year in CONFIG.backtest_years])) for name in scorecard.index]
scorecard[interval_label] = pd.Series(bootstrap_width)
mae_pairs = {name: persistence_and_recent_mean_mae(definition_series_all[name].dropna(), CONFIG.extended_backtest_years) for name in scorecard.index}
scorecard["erreur « reconduire l'an dernier »"] = [mae_pairs[name][0] for name in scorecard.index]
scorecard[recent_mean_label] = [mae_pairs[name][1] for name in scorecard.index]
scorecard[f"paires en {CONFIG.last_observed_year}"] = [len(last_year_subsets[name][0]) if name in last_year_subsets else int((leases.lease_year == CONFIG.last_observed_year).sum())
                                                       for name in scorecard.index]
scorecard["lisible face au TAL / Ontario"] = ["partiellement (mélange)", "non", "oui (renouvellements)", "non (marché libre)", "non"]
scorecard["usage de décision"] = ["lettres de renouvellement", "budget des revenus encaissés", "lettres de renouvellement", "prix de relocation", "aucun (mesure le mix)"]
scorecard.round(CONFIG.table_digits)

# %%
same_unit_names = list(DEFINITIONS)
mix_column = "écart tous immeubles vs panel fixe (points)"
naive_errors = scorecard.loc[same_unit_names, ["erreur « reconduire l'an dernier »", recent_mean_label]]
narrate(f"**Lecture du tableau.**\n"
        f"* **Mix :** les définitions à unité constante bougent d'au plus {fmt(scorecard.loc[same_unit_names, mix_column].max(), CONFIG.table_digits)} point entre "
        f"« tous immeubles » et « panel fixe » ; la médiane du portefeuille bouge de **{fmt(scorecard.loc[NAIVE_DEFINITION, mix_column])} points** : c'est le contre-exemple, à ne pas citer.\n"
        f"* **Bruit :** les intervalles des définitions à unité constante vont de {fmt(scorecard.loc[same_unit_names, interval_label].min(), CONFIG.table_digits)} à "
        f"{fmt(scorecard.loc[same_unit_names, interval_label].max(), CONFIG.table_digits)} point ; celui de la médiane du portefeuille vaut "
        f"{fmt(scorecard.loc[NAIVE_DEFINITION, interval_label])} points, en plus d'être trompeur.\n"
        f"* **Prévisibilité :** les références naïves se trompent de {fmt(naive_errors.min().min())} à {fmt(naive_errors.max().max())} points sur "
        f"{CONFIG.extended_backtest_years[0]}-{CONFIG.extended_backtest_years[-1]} selon la définition, parce que la série change de régime. Cela annonce la difficulté "
        "de la prévision (section 10) et justifie d'utiliser les règles et le marché plutôt que la seule tendance.\n"
        "* **Décision :** le **loyer effectif** répond à la question budgétaire (« combien de plus encaisserons-nous par unité ? ») ; le **contractuel** répond à la question "
        "réglementaire. Aucune des deux ne remplace l'autre, d'où la publication des deux.\n\n"
        "**Choix final.** Définition principale = **définition 2** (même unité, loyer effectif, médiane des paires, renouvellements et relocations). Définition secondaire = "
        "**définition 1** (même unité, loyer contractuel) avec les définitions 3 et 4 en détail. La définition 5 sert de contre-exemple pédagogique.")

# %% [markdown]
# ### 7b. Choix de forme, testés
# * **Année :** début de bail (`lease_start`) contre année de signature : le tableau ci-dessous mesure ce que le choix déplace.
# * **Agrégateur :** médiane (robuste, celle du départ) ; la moyenne pondérée par le loyer précédent (un budget additionne des dollars) est publiée en contrôle (section 4d).

# %%
sign_year_pairs = pairs.assign(sign_year=pairs.sign_date.dt.year)
year_attribution = pd.DataFrame({
    "année de début de bail (retenue)": pairs.groupby("lease_year").growth_effective_pct.median(),
    "année de signature": sign_year_pairs.groupby("sign_year").growth_effective_pct.median()}).loc[CONFIG.first_year:CONFIG.last_observed_year]
year_attribution["écart (points)"] = year_attribution.iloc[:, 1] - year_attribution.iloc[:, 0]
signing_lag_days = (leases.lease_start - leases.sign_date).dt.days
print(f"Délai signature → début de bail : médiane {signing_lag_days.median():.0f} jours, maximum {signing_lag_days.max():.0f} jours.")
year_attribution.round(CONFIG.table_digits)

# %%
assert definition_table.loc[CONFIG.last_observed_year, "2. même unité, effectif (RETENU)"] < definition_table.loc[CONFIG.last_observed_year, "1. même unité, contractuel"], "l'effectif doit être sous le contractuel en 2025"
assert scorecard.loc[NAIVE_DEFINITION, mix_column] > CONFIG.mix_sensitivity_ratio * scorecard.loc["1. même unité, contractuel", mix_column], "le contre-exemple doit être très sensible au mix"
print("Contrôles de la section 7 : OK")
