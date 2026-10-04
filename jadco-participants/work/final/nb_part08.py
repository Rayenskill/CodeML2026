# %% [markdown]
# ## 9. Contre-vérification : deux autres estimateurs de la même quantité
#
# La médiane des paires (section 4) est simple, mais elle n'utilise que les paires consécutives et un écart d'un an pour presque toutes. Deux estimateurs classiques, **indépendants** de la médiane, mesurent aussi la croissance « à même unité » : si les trois s'accordent, la mesure est solide ; sinon nous le disons.
#
# ### 9a. Indice de loyers répétés (méthode de Bailey-Muth-Nourse et de Case-Shiller)
# Pour chaque paire, `log(loyer_t2) − log(loyer_t1) = Σ_τ β_τ · D_τ + ε`, où `D_τ = +1` l'année du second bail, `−1` l'année du premier, `0` sinon. L'indice annuel est `exp(β_t − β_{t−1}) − 1` : **une seule estimation utilise toutes les paires**, y compris celles qui enjambent deux années (baux de 18 et 24 mois), que la médiane annuelle traite mal. Pondération de Case-Shiller : les écarts longs sont plus bruités, on pondère par l'inverse de la variance prédite en fonction de l'écart.
#
# **Cette étape ajoute une colonne** à `pairs` : `prev_lease_year`, l'année du bail précédent. Les années de l'indice vont du premier bail de l'extrait à la dernière année observée.

# %%
def repeat_rent_index(pair_frame, rent_column_pair, years):
    """Estime les effets d'année β (base = première année) par MCO puis MCP de Case-Shiller. Retourne la croissance annuelle (%)."""
    first, last = years[0], years[-1]
    frame = pair_frame[pair_frame.prev_lease_year.between(first, last) & pair_frame.lease_year.between(first, last)]
    y = np.log(frame[rent_column_pair[0]].to_numpy() / frame[rent_column_pair[1]].to_numpy())
    design = np.zeros((len(frame), len(years) - 1))
    for position, year in enumerate(years[1:]):
        design[:, position] += (frame.lease_year.to_numpy() == year)
        design[:, position] -= (frame.prev_lease_year.to_numpy() == year)
    beta_ols = np.linalg.lstsq(design, y, rcond=None)[0]
    squared = (y - design @ beta_ols) ** 2
    gaps = frame.gap_years.to_numpy()
    variance_fit = np.polyfit(gaps, squared, 1)
    predicted_variance = np.clip(np.polyval(variance_fit, gaps), squared.min() + CONFIG.variance_floor, None)
    sqrt_w = np.sqrt(1 / predicted_variance)[:, None]
    beta = np.linalg.lstsq(design * sqrt_w, y * sqrt_w[:, 0], rcond=None)[0]
    level = np.concatenate([[0.0], beta])
    return pd.Series(np.expm1(np.diff(level)) * 100, index=years[1:])


pairs["prev_lease_year"] = pairs.prev_lease_start.dt.year
index_years = list(range(int(leases.lease_year.min()), CONFIG.last_observed_year + 1))
repeat_contract = repeat_rent_index(pairs, ("rent_contract", "prev_rent_contract"), index_years)
repeat_effective = repeat_rent_index(pairs, ("rent_effective", "prev_rent_effective"), index_years)


def repeat_index_bootstrap(rent_pair, n_draws, rng):
    """Bootstrap par unité : on rééchantillonne les unités (les paires d'une même unité sont corrélées)."""
    unique_labels, inverse = np.unique(pairs.unit_label.to_numpy(), return_inverse=True)
    by_unit = [np.flatnonzero(inverse == k) for k in range(len(unique_labels))]
    draws = []
    for _ in range(n_draws):
        chosen = rng.integers(0, len(unique_labels), len(unique_labels))
        sample = pairs.iloc[np.concatenate([by_unit[k] for k in chosen])]
        draws.append(repeat_rent_index(sample, rent_pair, index_years))
    return pd.DataFrame(draws)


repeat_draws = repeat_index_bootstrap(("rent_effective", "prev_rent_effective"), CONFIG.n_repeat_bootstrap, np.random.default_rng(CONFIG.random_seed))
repeat_table = pd.DataFrame({
    "indice loyers répétés, contractuel": repeat_contract, "médiane des paires, contractuel": same_unit[("contractuel", "median")],
    "indice loyers répétés, effectif": repeat_effective, "médiane des paires, effectif": same_unit[("effectif", "median")],
    f"effectif, borne basse ({CONFIG.ci_low:.0f} %)": repeat_draws.quantile(CONFIG.ci_low / 100),
    f"effectif, borne haute ({CONFIG.ci_high:.0f} %)": repeat_draws.quantile(CONFIG.ci_high / 100),
}).loc[CONFIG.first_year:CONFIG.last_observed_year]
repeat_table.round(CONFIG.table_digits)

# %%
repeat_gap = (repeat_table["indice loyers répétés, effectif"] - repeat_table["médiane des paires, effectif"]).abs()
repeat_width = repeat_table.iloc[:, -1] - repeat_table.iloc[:, -2]
narrate(f"**Lecture.** L'indice de loyers répétés suit la médiane des paires de près (écart absolu de {fmt(repeat_gap.min(), CONFIG.table_digits)} à "
        f"{fmt(repeat_gap.max(), CONFIG.table_digits)} point selon l'année, même creux, même accélération) ; les écarts viennent de ce qu'il utilise des moyennes "
        "logarithmiques (sensibles aux valeurs extrêmes) et les paires à écart long. Les bandes bootstrap par unité ont une largeur de "
        f"{fmt(repeat_width.min(), CONFIG.table_digits)} à {fmt(repeat_width.max(), CONFIG.table_digits)} point : la **mesure historique est précise ; c'est la prévision qui est incertaine**.")

# %% [markdown]
# ### 9b. Modèle à effets fixes d'unité (régression hédonique « within »)
# `log(loyer_i,t) = α_unité + γ_année + δ_renouvellement + ε` : chaque unité est comparée à elle-même sur **tous** ses baux (pas seulement les paires consécutives), l'effet fixe d'unité absorbe tout ce qui est fixe dans un appartement (vue, plan, étage). Les effets d'année `γ` forment un indice de qualité constante ; la différence `γ_t − γ_{t−1}` est la croissance.

# %%
fixed_effects_sample = leases[leases.lease_year.between(index_years[0], CONFIG.last_observed_year)].copy()


def within_year_effects(frame, rent_column):
    """Régression within : année + renouvellement, après suppression de la moyenne par unité (effet fixe d'unité)."""
    data = frame.copy()
    data["log_rent"] = np.log(data[rent_column])
    year_dummies = pd.get_dummies(data.lease_year, prefix="y", drop_first=True).astype(float)
    design = pd.concat([year_dummies, data[["is_renewal"]].astype(float)], axis=1)
    design["unit_label"] = (data.prop_code + "|" + data.unit_code).to_numpy()
    demeaned = design.drop(columns="unit_label") - design.groupby("unit_label").transform("mean")
    demeaned_y = data.log_rent - data.log_rent.groupby(design.unit_label).transform("mean")
    fit = sm.OLS(demeaned_y, demeaned).fit()
    effects = pd.concat([pd.Series({f"y_{index_years[0]}": 0.0}), fit.params.drop("is_renewal")])
    return pd.Series(np.expm1(np.diff(effects.to_numpy())) * 100, index=index_years[1:]), fit.params["is_renewal"]


within_effective, renewal_premium_effective = within_year_effects(fixed_effects_sample, "rent_effective")
triangulation = pd.DataFrame({
    "médiane des paires": same_unit[("effectif", "median")], "loyers répétés": repeat_effective, "effets fixes d'unité": within_effective,
}).loc[CONFIG.first_year:CONFIG.last_observed_year]
triangulation["écart max entre méthodes (points)"] = triangulation.max(axis=1) - triangulation.min(axis=1)
print(f"Effet moyen « renouvellement » à unité et année données : {np.expm1(renewal_premium_effective) * 100:+.2f} % (loyer effectif)")
triangulation.round(CONFIG.table_digits)

# %%
narrate(f"**Lecture.** Trois méthodes indépendantes (médiane de paires, loyers répétés, effets fixes d'unité) donnent des croissances effectives proches année après année : "
        f"l'écart maximal entre elles va de {fmt(triangulation.iloc[:, -1].min(), CONFIG.table_digits)} à {fmt(triangulation.iloc[:, -1].max(), CONFIG.table_digits)} point. "
        "La croissance historique à unité constante est bien mesurée. L'incertitude ne vient donc pas de la mesure, mais de la **dynamique** (changements de régime), "
        "ce qui explique pourquoi la prévision de la section 10 reste incertaine malgré une mesure précise.")

# %%
assert (triangulation["écart max entre méthodes (points)"] < CONFIG.max_method_disagreement_points).all(), "les trois méthodes doivent s'accorder"
assert abs(repeat_table.loc[CONFIG.last_observed_year, "indice loyers répétés, contractuel"] - same_unit.loc[CONFIG.last_observed_year, ("contractuel", "median")]) < CONFIG.max_method_disagreement_points
print("Contrôles de la section 9 : OK")
