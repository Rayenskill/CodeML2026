# %% [markdown]
# ## 3. Pourquoi la médiane annuelle mesure surtout le mix
#
# **Étape sans modification des données sources** : on calcule des tables d'analyse (`naive`, `mix_profile`, `entry_year`). Le réflexe est le loyer médian par année de début de bail. Nous le reproduisons, puis nous montrons trois fois, de la plus simple à la plus rigoureuse, que ce nombre reflète la *composition* du portefeuille et non les prix : (a) un **panel fixe** de propriétés, (b) une **décomposition exacte** de la variation de loyer moyen, (c) un **indice de prix de Fisher** sur des segments comparables. Le segment de base est le **code de propriété** (`prop_code`), parce que le nom d'immeuble « Saint-Elzéar » regroupe trois propriétés distinctes.

# %%
analysis_window = leases[leases.lease_year.between(CONFIG.first_year, CONFIG.last_observed_year)]

naive = analysis_window.groupby("lease_year").agg(leases=("rent_contract", "size"), median_rent=("rent_contract", "median"))
naive["yoy_pct"] = naive.median_rent.pct_change() * 100

mix_profile = analysis_window.groupby("lease_year").agg(
    median_sqft=("sqft", "median"),
    pct_2bed_plus=("beds", lambda beds: (beds >= 2).mean() * 100),
    properties=("prop_code", "nunique"))
mix_profile["median_rent_per_sqft"] = (analysis_window.assign(rent_psf=analysis_window.rent_contract / analysis_window.sqft)
                                       .groupby("lease_year").rent_psf.median())
entry_year = leases.groupby("prop_code").lease_year.min().sort_values()
print("Année du premier bail par code de propriété (mise en service) :")
print(entry_year.to_dict())
pd.concat([naive, mix_profile], axis=1)

# %%
naive_peak_year = int(naive.yoy_pct.idxmax())
entrants_by_year = {year: ", ".join(f"`{code}`" for code in codes.index) for year, codes in entry_year[entry_year > CONFIG.first_year].groupby(entry_year)}
narrate(f"**Lecture.** La médiane naïve culmine à **{fmt(naive.yoy_pct[naive_peak_year], signed=True)} % en {naive_peak_year}** (le chiffre de l'énoncé). "
        f"La même année, le loyer médian au pied carré passe de {fmt(mix_profile.median_rent_per_sqft[naive_peak_year - 1], CONFIG.table_digits)} à "
        f"{fmt(mix_profile.median_rent_per_sqft[naive_peak_year], CONFIG.table_digits)} $ et la part des 2 chambres et plus de "
        f"{fmt(mix_profile.pct_2bed_plus[naive_peak_year - 1])} % à {fmt(mix_profile.pct_2bed_plus[naive_peak_year])} % : des propriétés entrent dans l'extrait "
        f"({' ; '.join(f'{year} : {codes}' for year, codes in entrants_by_year.items())}). Presque aucun locataire existant ne paie cette hausse, comme le montrent les trois tests suivants.")

# %% [markdown]
# ### 3a. Panel fixe : mêmes propriétés chaque année
# On ne garde que les codes de propriété déjà en service avant les mises en service de 2023-2024 (`panel_entry_year`). Si la hausse de 2023 vient du mix, elle doit disparaître.

# %%
panel_codes = entry_year[entry_year <= CONFIG.panel_entry_year].index
panel_median = (analysis_window[analysis_window.prop_code.isin(panel_codes)]
                .groupby("lease_year").rent_contract.median())

fig, axes = plt.subplots(1, 2, figsize=STYLE.figsize_wide)
axes[0].plot(naive.index, naive.median_rent, "o-", color=PALETTE["naive"], label="tous les baux")
axes[0].plot(panel_median.index, panel_median, "o-", color=PALETTE["contract"], label=f"panel fixe (propriétés ≤ {CONFIG.panel_entry_year})")
axes[0].set(title="Loyer médian des baux débutant l'année t", ylabel="$ par mois")
axes[0].legend()
share_by_property = (analysis_window.groupby(["lease_year", "prop_code"]).size().unstack(fill_value=0))
(share_by_property.div(share_by_property.sum(axis=1), axis=0) * 100).plot.area(ax=axes[1], alpha=STYLE.area_alpha)
axes[1].set(title="Part des baux par code de propriété (%)", ylabel="%")
axes[1].legend(fontsize=STYLE.small_fontsize, loc="upper left")
plt.tight_layout()
plt.show()
panel_yoy = panel_median.pct_change() * 100
pd.DataFrame({"naïf (tous les baux)": naive.yoy_pct, "panel fixe": panel_yoy}).loc[CONFIG.first_year + 1:]

# %%
narrate(f"**Conclusion 3a.** À composition de propriétés constante, {naive_peak_year} passe de {fmt(naive.yoy_pct[naive_peak_year], signed=True)} % "
        f"à {fmt(panel_yoy[naive_peak_year], signed=True)} %. Les deux courbes se séparent exactement quand les nouvelles propriétés entrent.")

# %% [markdown]
# ### 3b. Décomposition exacte de la variation du loyer moyen
# Pour chaque année `t`, la variation du loyer *moyen* des baux se découpe en trois morceaux qui s'additionnent **exactement** :
# 1. **Prix à segment constant** : les segments (propriété × nombre de chambres) présents deux années de suite, pondérés par leur poids de l'année précédente ;
# 2. **Recomposition interne** : le déplacement des poids entre segments de propriétés déjà en service ;
# 3. **Entrée de nouvelles propriétés** : ce que change l'arrivée d'un code de propriété qui n'avait aucun bail l'année précédente.
#
# C'est une décomposition par *shift-share* (écart de prix à l'intérieur des segments contre écart de composition).

# %%
SEGMENT_KEY = ["prop_code", "beds"]


def decompose_mean_change(frame, year):
    """Décompose la variation du loyer moyen entre year-1 et year (en % du loyer moyen de year-1)."""
    previous = frame[frame.lease_year == year - 1]
    current = frame[frame.lease_year == year]
    mean_previous = previous.rent_contract.mean()
    total_change = current.rent_contract.mean() - mean_previous
    continuing = set(previous.prop_code) & set(current.prop_code)
    previous_c = previous[previous.prop_code.isin(continuing)]
    current_c = current[current.prop_code.isin(continuing)]
    change_continuing = current_c.rent_contract.mean() - previous_c.rent_contract.mean()

    def segment_table(part):
        table = part.groupby(SEGMENT_KEY).rent_contract.agg(["size", "mean"])
        table["weight"] = table["size"] / table["size"].sum()
        return table

    joined = segment_table(previous_c).join(segment_table(current_c), lsuffix="_prev", rsuffix="_curr", how="inner")
    within = (joined.weight_prev * (joined.mean_curr - joined.mean_prev)).sum()
    parts = {"within": within, "internal_mix": change_continuing - within, "entry": total_change - change_continuing}
    parts = {name: value / mean_previous * 100 for name, value in parts.items()}
    parts["total"] = total_change / mean_previous * 100
    return parts


decomposition = pd.DataFrame({year: decompose_mean_change(analysis_window, year)
                              for year in range(CONFIG.first_year + 1, CONFIG.last_observed_year + 1)}).T
decomposition.columns = ["prix à segment constant", "recomposition interne", "entrée de nouvelles propriétés", "total (loyer moyen)"]
COMPONENT_COLUMNS = list(decomposition.columns[:-1])      # les trois morceaux ; la dernière colonne est le total
TOTAL_COLUMN = decomposition.columns[-1]
assert np.allclose(decomposition[COMPONENT_COLUMNS].sum(axis=1), decomposition[TOTAL_COLUMN]), "la décomposition doit se refermer"
decomposition

# %%
fig, ax = plt.subplots(figsize=STYLE.figsize_wide)
decomposition[COMPONENT_COLUMNS].plot.bar(stacked=True, ax=ax, color=[PALETTE["contract"], PALETTE["rule"], PALETTE["effective"]])
ax.plot(range(len(decomposition)), decomposition[TOTAL_COLUMN].to_numpy(), "ko", label="total")
ax.axhline(0, color="black", lw=STYLE.line_width / 2)
ax.set(title="D'où vient la variation du loyer moyen ? (points de %, la somme est exacte)", ylabel="points de %", xlabel="année de début de bail")
ax.legend(fontsize=STYLE.legend_fontsize)
plt.tight_layout()
plt.show()

# %%
peak_parts = decomposition.loc[naive_peak_year]
last_parts = decomposition.loc[CONFIG.last_observed_year]
narrate(f"**Conclusion 3b.** En {naive_peak_year}, sur {fmt(peak_parts[TOTAL_COLUMN], signed=True)} points de variation du loyer moyen, "
        f"l'entrée de nouvelles propriétés en explique **{fmt(peak_parts['entrée de nouvelles propriétés'], signed=True)}** et le prix à segment constant "
        f"seulement {fmt(peak_parts['prix à segment constant'], signed=True)}. En {CONFIG.last_observed_year}, le prix à segment constant vaut "
        f"{fmt(last_parts['prix à segment constant'], signed=True)} point(s) et la recomposition interne {fmt(last_parts['recomposition interne'], signed=True)}. "
        "Ces morceaux se somment exactement : la hausse naïve est une affaire de composition.")

# %% [markdown]
# ### 3c. Indice de Fisher sur segments appariés
# Contre-vérification par la théorie des indices : sur les segments (propriété × chambres) présents deux années de suite, indice de Laspeyres (poids de l'année précédente), de Paasche (poids courants) et leur moyenne géométrique, l'indice de **Fisher**. Les segments nouveaux sont exclus : le prix ne se compare qu'à ce qui existait.

# %%
def fisher_growth(frame, year):
    previous = frame[frame.lease_year == year - 1].groupby(SEGMENT_KEY).rent_contract.agg(["size", "mean"])
    current = frame[frame.lease_year == year].groupby(SEGMENT_KEY).rent_contract.agg(["size", "mean"])
    matched = previous.join(current, lsuffix="_prev", rsuffix="_curr", how="inner")
    laspeyres = (matched.size_prev * matched.mean_curr).sum() / (matched.size_prev * matched.mean_prev).sum()
    paasche = (matched.size_curr * matched.mean_curr).sum() / (matched.size_curr * matched.mean_prev).sum()
    return (np.sqrt(laspeyres * paasche) - 1) * 100


fisher = pd.Series({year: fisher_growth(analysis_window, year)
                    for year in range(CONFIG.first_year + 1, CONFIG.last_observed_year + 1)}, name="fisher_pct")
pd.DataFrame({"médiane naïve": naive.yoy_pct, "Fisher (segments appariés)": fisher}).loc[CONFIG.first_year + 1:]

# %%
narrate(f"**Conclusion 3c.** L'indice de Fisher donne {fmt(fisher[naive_peak_year], signed=True)} % en {naive_peak_year} "
        f"(contre {fmt(naive.yoy_pct[naive_peak_year], signed=True)} % pour la médiane naïve) et {fmt(fisher[CONFIG.last_observed_year], signed=True)} % "
        f"en {CONFIG.last_observed_year} : il retrouve la bonne histoire sans comparer bail à bail. La section 4 montre que la méthode « même unité contre elle-même » "
        "donne des chiffres du même ordre par une autre voie : trois méthodes indépendantes, une même conclusion.")
