# %% [markdown]
# ## 4. Comparer chaque unité à elle-même
#
# **Étape qui crée une nouvelle table de données** : `pairs`, une ligne par bail et son bail *précédent* dans la **même unité**. La clé d'unité est `prop_code + unit_code` (équivalente à `unit_id`, vérifié en section 2), **jamais** `site + unit_code`.
#
# Plutôt que de filtrer en silence, le moteur construit **toutes** les paires consécutives, puis *étiquette* chaque raison d'invalidité. On peut ainsi compter ce qu'on écarte et tester si le filtre change la réponse.
#
# **Annualisation.** Pour un écart de `gap` années entre deux débuts de bail : `croissance = (loyer / loyer_précédent) ** (1 / gap) - 1`. Les baux de 12 mois ont un écart voisin de 1 an (l'annualisation ne change presque rien) ; les baux de 18 et 24 mois, si.

# %%
def annualised_growth_pct(rent, previous_rent, gap_years):
    """Variation annualisée en % (composée) entre deux loyers séparés de gap_years années."""
    return ((rent / previous_rent) ** (1 / gap_years) - 1) * 100


def build_pairs(frame, key=UNIT_KEY):
    """Une ligne par bail ayant un bail précédent dans la même unité (selon `key`), avec les motifs d'invalidité."""
    ordered = frame.sort_values(key + ["lease_start"])
    grouped = ordered.groupby(key)
    pairs = ordered.copy()
    carried = ["lease_start", "rent_contract", "rent_effective", "sqft", "beds", "term_months", "unit_id", "prop_code", "is_renewal", "drag"]
    for column in carried:
        pairs[f"prev_{column}"] = grouped[column].shift(1)
    pairs = pairs.dropna(subset=["prev_lease_start"]).copy()
    pairs["gap_years"] = (pairs.lease_start - pairs.prev_lease_start).dt.days / CONFIG.days_per_year
    # motifs d'invalidité (une colonne booléenne chacun)
    pairs["flag_gap_too_short"] = pairs.gap_years <= CONFIG.pair_gap_min_years
    pairs["flag_gap_too_long"] = pairs.gap_years >= CONFIG.pair_gap_max_years
    pairs["flag_nonpositive_rent"] = (pairs.prev_rent_contract <= 0) | (pairs.rent_contract <= 0) | (pairs.prev_rent_effective <= 0) | (pairs.rent_effective <= 0)
    pairs["flag_other_apartment"] = (pairs.prev_unit_id != pairs.unit_id) | (pairs.prev_sqft != pairs.sqft) | (pairs.prev_beds != pairs.beds)
    pairs["flag_term_changed"] = pairs.prev_term_months != pairs.term_months
    pairs["is_valid"] = ~(pairs.flag_gap_too_short | pairs.flag_gap_too_long | pairs.flag_nonpositive_rent)
    positive = pairs.gap_years > 0
    for kind in ("contract", "effective"):
        pairs[f"growth_{kind}_pct"] = np.where(
            positive & ~pairs.flag_nonpositive_rent,
            annualised_growth_pct(pairs[f"rent_{kind}"], pairs[f"prev_rent_{kind}"].where(~pairs.flag_nonpositive_rent, 1), pairs.gap_years.where(positive, 1)),
            np.nan)
    pairs["growth_raw_contract_pct"] = (pairs.rent_contract / pairs.prev_rent_contract - 1) * 100
    return pairs


pairs_all = build_pairs(leases)
pairs = pairs_all[pairs_all.is_valid].copy()
print(f"{len(pairs_all):,} paires consécutives, dont {len(pairs):,} valides sur {pairs.groupby(UNIT_KEY).ngroups:,} unités")
assert len(pairs_all) == len(leases) - leases.groupby(UNIT_KEY).ngroups, "toute unité a (n baux - 1) paires"

# %% [markdown]
# ### 4a. Entonnoir de validité : que retire-t-on, et pourquoi ?
# Chaque raison est comptée séparément (une paire peut en cumuler plusieurs). Aucune paire n'est perdue sans être comptée : la somme des paires valides et des paires invalides retombe sur le total.

# %%
funnel_labels = {
    "all": "toutes les paires consécutives",
    "short": f"écart ≤ {fmt(CONFIG.pair_gap_min_years)} an (même bail enregistré deux fois ?)",
    "long": f"écart ≥ {fmt(CONFIG.pair_gap_max_years)} ans (non comparable)",
    "nonpositive": "loyer nul ou négatif",
    "dropped": "= paires écartées (union)",
    "valid": "= paires valides (filtre du notebook de départ)",
    "other_apartment": "pour information : appartement différent (sqft/chambres/id)",
    "term_changed": "pour information : durée de bail différente"}
funnel = pd.DataFrame({"paires": {
    funnel_labels["all"]: len(pairs_all), funnel_labels["short"]: pairs_all.flag_gap_too_short.sum(), funnel_labels["long"]: pairs_all.flag_gap_too_long.sum(),
    funnel_labels["nonpositive"]: pairs_all.flag_nonpositive_rent.sum(), funnel_labels["dropped"]: (~pairs_all.is_valid).sum(),
    funnel_labels["valid"]: pairs_all.is_valid.sum(), funnel_labels["other_apartment"]: pairs_all.flag_other_apartment.sum(),
    funnel_labels["term_changed"]: pairs_all.flag_term_changed.sum()}})
funnel["part (%)"] = funnel.paires / len(pairs_all) * 100
print(funnel.round(CONFIG.prose_digits).to_string())
assert funnel.loc[funnel_labels["dropped"], "paires"] + funnel.loc[funnel_labels["valid"], "paires"] == len(pairs_all)
assert pairs_all.flag_other_apartment.sum() == 0, "sur la bonne clé, une unité garde ses attributs d'un bail à l'autre"

# %%
narrate(f"**Lecture.** Le filtre du notebook de départ (écart strictement entre {fmt(CONFIG.pair_gap_min_years)} et {fmt(CONFIG.pair_gap_max_years)} ans) "
        f"écarte {int((~pairs_all.is_valid).sum())} paires sur {fmt(len(pairs_all), 0)} ({fmt((~pairs_all.is_valid).mean() * 100)} %) ; "
        f"**aucune paire sur la bonne clé ne change d'appartement** (même `unit_id`, mêmes pieds carrés, même nombre de chambres). La clé `prop_code + unit_code` est donc sûre.")

# %% [markdown]
# ### 4b. La mauvaise clé échoue : démonstration
# Que se passe-t-il si l'on apparie sur `site + unit_code` ? Le site « saint-elzear » regroupe trois codes de propriété (`stelz1`, `stelz2`, `stelz3`) dont les numéros d'unité se chevauchent : on crée de fausses paires entre deux appartements différents.

# %%
def percentile_spread(values):
    """Écart entre les percentiles haut et bas de l'intervalle de référence : mesure de dispersion robuste."""
    return values.quantile(CONFIG.ci_high / 100) - values.quantile(CONFIG.ci_low / 100)


wrong_pairs = build_pairs(leases, key=["site", "unit_code"])
wrong_valid = wrong_pairs[wrong_pairs.is_valid]
fake_pairs = wrong_valid[wrong_valid.prev_unit_id != wrong_valid.unit_id]
print(f"Avec la clé site + code d'unité : {len(wrong_valid):,} paires 'valides' dont {len(fake_pairs):,} relient deux appartements DIFFÉRENTS "
      f"({len(fake_pairs) / len(wrong_valid):.1%}).")
fake_by_codes = fake_pairs.groupby(["prev_prop_code", "prop_code"]).size().sort_values(ascending=False)
print("Codes de propriété concernés :", fake_by_codes.to_dict())
genuine_pairs_kept = len(wrong_valid) - len(fake_pairs)
genuine_pairs_lost = len(pairs) - genuine_pairs_kept
print(f"Paires authentiques perdues : {genuine_pairs_lost:,} sur {len(pairs):,} (une fausse paire s'intercale entre deux baux de la même unité).")
spread_label = f"p{CONFIG.ci_high:.0f} − p{CONFIG.ci_low:.0f}"
wrong_vs_right = pd.DataFrame({
    "bonne clé : médiane": pairs.groupby("lease_year").growth_contract_pct.median(),
    "mauvaise clé : médiane": wrong_valid.groupby("lease_year").growth_contract_pct.median(),
    f"bonne clé : {spread_label}": pairs.groupby("lease_year").growth_contract_pct.apply(percentile_spread),
    f"mauvaise clé : {spread_label}": wrong_valid.groupby("lease_year").growth_contract_pct.apply(percentile_spread),
}).loc[CONFIG.first_year:]
wrong_vs_right["biais de la médiane (points)"] = wrong_vs_right["mauvaise clé : médiane"] - wrong_vs_right["bonne clé : médiane"]
by_code_right = pairs[pairs.lease_year == CONFIG.last_observed_year].groupby("prop_code").growth_contract_pct.median()
by_code_wrong = wrong_valid[wrong_valid.lease_year == CONFIG.last_observed_year].groupby("prop_code").growth_contract_pct.median()
wrong_by_code = pd.DataFrame({"bonne clé": by_code_right, "mauvaise clé": by_code_wrong}).dropna()
wrong_by_code["biais (points)"] = wrong_by_code["mauvaise clé"] - wrong_by_code["bonne clé"]
display(wrong_vs_right.round(CONFIG.table_digits))
display(wrong_by_code.round(CONFIG.table_digits))

# %%
reused_codes = len(set(leases[leases.prop_code == "stelz3"].unit_code) & set(leases[leases.prop_code == "stelz1"].unit_code))
first_spread_year = wrong_vs_right.index.min()
most_biased_code = wrong_by_code["biais (points)"].abs().idxmax()
narrate(f"**Lecture (chiffres mesurés ci-dessus).** Apparier sur `site + unit_code` :\n"
        f"* fabrique des **fausses paires** entre deux appartements différents ({fmt(len(fake_pairs) / len(wrong_valid) * 100)} % des paires « valides », "
        f"surtout `{fake_by_codes.index[0][0]}` ↔ `{fake_by_codes.index[0][1]}` ; `stelz3` réutilise {reused_codes} codes d'unité de `stelz1`) ;\n"
        f"* **détruit {fmt(genuine_pairs_lost, 0)} vraies paires** : la fausse paire s'intercale et raccourcit l'écart au point de faire sauter le filtre ;\n"
        f"* **gonfle la dispersion** (écart {spread_label} de {fmt(wrong_vs_right[f'mauvaise clé : {spread_label}'].iloc[0])} contre "
        f"{fmt(wrong_vs_right[f'bonne clé : {spread_label}'].iloc[0])} points en {first_spread_year}) ;\n"
        f"* biaise les propriétés touchées (jusqu'à {fmt(wrong_by_code.loc[most_biased_code, 'biais (points)'], CONFIG.table_digits, signed=True)} point pour `{most_biased_code}` "
        f"en {CONFIG.last_observed_year}), alors que la **médiane du portefeuille bouge peu** (au plus {fmt(wrong_vs_right['biais de la médiane (points)'].abs().max(), CONFIG.table_digits)} point) "
        "parce que les erreurs se compensent en gros.\n\n"
        "Le danger de la mauvaise clé n'est donc pas un biais spectaculaire de la médiane globale : c'est une mesure **bruitée et fausse propriété par propriété**, "
        "inutilisable pour une prévision par immeuble. Notre moteur évite le piège et le démontre.")

# %% [markdown]
# ### 4c. Hausse à unité constante, par année
# Les médianes ci-dessous reprennent la définition de départ (contractuel, effectif) et ajoutent la moyenne pondérée par le loyer précédent, un contrôle de robustesse (un budget additionne des dollars). Le compte de paires par année justifie le choix de `first_year` : avant, les années comptent trop peu de paires.

# %%
def summarise_growth(frame, value_column, weight_column):
    """Statistiques par année de début de bail : n, médiane, moyenne, moyenne pondérée par le loyer précédent."""
    def per_year(group):
        weights = group[weight_column]
        return pd.Series({"pairs": len(group), "median": group[value_column].median(), "mean": group[value_column].mean(),
                          "weighted_mean": np.average(group[value_column], weights=weights)})
    return frame.groupby("lease_year").apply(per_year, include_groups=False)


same_unit_all_years = pd.concat({
    "contractuel": summarise_growth(pairs, "growth_contract_pct", "prev_rent_contract"),
    "effectif": summarise_growth(pairs, "growth_effective_pct", "prev_rent_effective"),
}, axis=1)
print("Paires valides par année (toutes années) :", same_unit_all_years[("contractuel", "pairs")].astype(int).to_dict())
same_unit = same_unit_all_years.loc[CONFIG.first_year:CONFIG.last_observed_year]
same_unit.round(CONFIG.table_digits)

# %%
fig, ax = plt.subplots(figsize=STYLE.figsize_wide)
years_shown = list(range(CONFIG.extended_backtest_years[0], CONFIG.last_observed_year + 1))
ax.plot(years_shown, naive.loc[years_shown, "yoy_pct"], "o--", color=PALETTE["naive"], lw=STYLE.line_width, label="naïf : médiane de tous les baux")
ax.plot(years_shown, fisher.loc[years_shown], "s:", color=PALETTE["market"], lw=STYLE.line_width, label="indice de Fisher (segments appariés)")
ax.plot(years_shown, same_unit.loc[years_shown, ("contractuel", "median")], "o-", color=PALETTE["contract"], lw=STYLE.line_width, label="même unité, loyer contractuel")
ax.plot(years_shown, same_unit.loc[years_shown, ("effectif", "median")], "o-", color=PALETTE["effective"], lw=STYLE.line_width, label="même unité, loyer effectif")
ax.axhline(0, color="grey", lw=STYLE.line_width / 2)
ax.set(ylabel="croissance annuelle des loyers (%)", title="La médiane naïve mesure surtout la composition")
ax.legend(fontsize=STYLE.legend_fontsize)
plt.tight_layout()
plt.show()

# %%
contract_median = same_unit[("contractuel", "median")]
effective_median = same_unit[("effectif", "median")]
trough_year = int(contract_median.loc[CONFIG.extended_backtest_years[0]:].idxmin())
narrate(f"**Lecture.** Les trois méthodes sans effet de mix (Fisher, même unité contractuel, même unité effectif) racontent la même histoire : "
        f"un creux en {trough_year} ({fmt(contract_median[trough_year])} % au contractuel), puis une accélération en {CONFIG.last_observed_year} "
        f"pour le contractuel ({fmt(contract_median[CONFIG.last_observed_year])} %) que l'effectif n'accompagne qu'en partie "
        f"({fmt(effective_median[CONFIG.last_observed_year])} %, voir section 5). La médiane naïve invente un pic en {naive_peak_year}.")

# %% [markdown]
# ### 4d. Sensibilité : le résultat dépend-il des filtres ?
# On fait varier trois choix (fenêtre d'écart, traitement des extrêmes, agrégateur). Si la hausse de 2025 bouge peu, le résultat ne vient pas d'un filtre arbitraire ; sinon on dit lequel.

# %%
gap_windows = {f"{fmt(low, CONFIG.table_digits)}–{fmt(high, CONFIG.table_digits)} ans": (low, high)
               for low, high in ((CONFIG.pair_gap_min_years, CONFIG.pair_gap_max_years),) + CONFIG.sensitivity_gap_windows}
outlier_policies = {"none": "aucun", "winsorise": f"winsorisation {fmt(CONFIG.winsor_share * 100)} %",
                    "trim": f"rognage {fmt(CONFIG.outlier_tail_share * 100)} % de chaque côté"}
aggregators = ("médiane", "moyenne", "moyenne pondérée")


def apply_outlier_policy(values, policy):
    if policy == "none":
        return values
    if policy == "winsorise":
        low, high = values.quantile([CONFIG.winsor_share, 1 - CONFIG.winsor_share])
        return values.clip(low, high)
    low, high = values.quantile([CONFIG.outlier_tail_share, 1 - CONFIG.outlier_tail_share])
    return values.where(values.between(low, high))


def sensitivity_table(rent_kind, years):
    rows = []
    for window_name, (gap_low, gap_high) in gap_windows.items():
        base = pairs_all[pairs_all.gap_years.gt(gap_low) & pairs_all.gap_years.lt(gap_high) & ~pairs_all.flag_nonpositive_rent]
        for policy, policy_label in outlier_policies.items():
            values = apply_outlier_policy(base[f"growth_{rent_kind}_pct"], policy)
            for aggregator in aggregators:
                row = {"fenêtre": window_name, "extrêmes": policy_label, "agrégateur": aggregator}
                for year in years:
                    sample = values[base.lease_year == year].dropna()
                    weights = base.loc[sample.index, f"prev_rent_{rent_kind}"]
                    row[year] = {"médiane": sample.median(), "moyenne": sample.mean(), "moyenne pondérée": np.average(sample, weights=weights)}[aggregator]
                rows.append(row)
    return pd.DataFrame(rows)


sensitivity_contract = sensitivity_table("contract", CONFIG.backtest_years)
sensitivity_effective = sensitivity_table("effective", CONFIG.backtest_years)
n_combinations = len(sensitivity_contract)
last_year_range = pd.DataFrame({
    "contractuel": [sensitivity_contract[CONFIG.last_observed_year].min(), sensitivity_contract[CONFIG.last_observed_year].max()],
    "effectif": [sensitivity_effective[CONFIG.last_observed_year].min(), sensitivity_effective[CONFIG.last_observed_year].max()]},
    index=[f"minimum sur les {n_combinations} combinaisons", f"maximum sur les {n_combinations} combinaisons"])
print(f"Amplitude de la hausse {CONFIG.last_observed_year} selon les filtres :")
print(last_year_range.round(CONFIG.table_digits))
sensitivity_contract.round(CONFIG.table_digits)

# %%
previous_years_max = sensitivity_contract[list(CONFIG.backtest_years[:-1])].max().max()
gap_range = (sensitivity_contract[CONFIG.last_observed_year] - sensitivity_effective[CONFIG.last_observed_year])
aggregator_shift = (sensitivity_contract[sensitivity_contract.agrégateur == "moyenne pondérée"][CONFIG.last_observed_year].mean()
                    - sensitivity_contract[sensitivity_contract.agrégateur == "médiane"][CONFIG.last_observed_year].mean())
narrate(f"**Lecture.** Sur les {n_combinations} combinaisons, la hausse contractuelle {CONFIG.last_observed_year} va de "
        f"{fmt(last_year_range.iloc[0, 0])} à {fmt(last_year_range.iloc[1, 0])} %, toujours au-dessus du maximum des années de comparaison "
        f"({CONFIG.backtest_years[0]}-{CONFIG.backtest_years[-2]} : {fmt(previous_years_max)} %) ; l'écart contractuel − effectif reste entre {fmt(gap_range.min())} et {fmt(gap_range.max())} points. "
        f"L'agrégateur déplace le niveau d'environ {fmt(aggregator_shift, signed=True)} point (moyenne pondérée contre médiane) : nous publions la médiane "
        "(comparable au notebook de départ) et la moyenne pondérée comme contrôle.")

# %%
# Contrôles de validité (parité avec le notebook de départ) --------------------------------------------------
starter_style = pairs_all[pairs_all.gap_years.between(CONFIG.pair_gap_min_years, CONFIG.pair_gap_max_years, inclusive="neither")]
assert len(starter_style) == len(pairs), "notre filtre = celui du notebook de départ"
assert abs(same_unit.loc[CONFIG.last_observed_year, ("contractuel", "median")] - CONFIG.starter_contract_median_2025) < CONFIG.check_tolerance, "médiane contractuelle 2025 du départ"
assert abs(same_unit.loc[CONFIG.last_observed_year, ("effectif", "median")] - CONFIG.starter_effective_median_2025) < CONFIG.check_tolerance, "médiane effective 2025 du départ"
assert len(fake_pairs) > 0, "la mauvaise clé doit produire de fausses paires"
assert last_year_range.iloc[0, 0] > previous_years_max, "la hausse contractuelle 2025 dépasse les années précédentes quel que soit le filtre"
print("Contrôles de la section 4 : OK")
