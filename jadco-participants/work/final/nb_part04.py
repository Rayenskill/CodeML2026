# %% [markdown]
# ## 5. Loyer contractuel n'est pas loyer perçu
#
# `rent_contract` est le loyer inscrit au bail ; `rent_effective` est ce que `Équinoxe` perçoit réellement après les concessions. Cette section (a) **vérifie ce que sont les lignes de concession** (en particulier `PromoPay`), (b) **reconstruit** `rent_effective` à partir de ces lignes pour décider auquel des deux loyers se fier, (c) **explique pourquoi la croissance contractuelle et la croissance effective divergent à partir de 2024**.
#
# ### 5a. Que sont les codes de concession ?
# Pour un montant `amount` (négatif : un rabais) couvrant `months_covered` mois, l'équivalent mensuel sur la durée du bail est `−amount × months_covered / term_months`. Pour `PromoPay`, l'énoncé annonce un frais *mensuel*. Nous mesurons ce que la colonne contient réellement avant de la modéliser.
#
# **Étape qui crée une table** : `linked`, chaque ligne de concession rattachée à son bail, avec la colonne `amount_over_monthly_rent`.

# %%
concessions = concessions.reset_index(drop=True)
linked = link_concession_lines(concessions, leases, CONFIG.concession_link_grace_days)
linked["amount_over_monthly_rent"] = -linked.amount / linked.rent_contract

code_profile = linked.groupby("charge_code").agg(
    lines=("amount", "size"), median_amount=("amount", "median"), median_months_covered=("months_covered", "median"),
    median_amount_over_monthly_rent=("amount_over_monthly_rent", "median")).sort_values("lines", ascending=False)
print(f"{len(linked):,} lignes de concession rattachées à un bail sur {len(concessions):,} ({len(linked) / len(concessions):.1%}).")
code_profile

# %%
promo_profile = code_profile.loc["PromoPay"]
other_codes = code_profile.drop(index="PromoPay")
narrate(f"**Lecture.** `PromoPay` couvre en médiane **{fmt(promo_profile.median_months_covered, 0)} mois** pour un montant égal à "
        f"**{fmt(promo_profile.median_amount_over_monthly_rent, CONFIG.table_digits)} fois le loyer mensuel** : c'est un **forfait unique**, pas un frais mensuel. "
        f"Les autres codes ({', '.join(f'`{code}`' for code in other_codes.index)}) sont de petits montants mensuels "
        f"({fmt(-other_codes.median_amount.max(), 0)} à {fmt(-other_codes.median_amount.min(), 0)} $ en médiane) sur toute la durée du bail. "
        f"Rattachement : une ligne appartient au *dernier bail commencé au plus tard {CONFIG.concession_link_grace_days} jours après son début* "
        "(une concession peut débuter un peu avant le bail ; le test de la section 5b montre que ce délai maximise la concordance).")

# %% [markdown]
# ### 5b. Reconstruire le loyer effectif
# On étale chaque ligne sur la durée du bail (`PromoPay` en forfait unique divisé par la durée) et on compare à l'écart réel `rent_contract − rent_effective`. Puis on fait varier le délai de rattachement : la valeur retenue dans `CONFIG` doit être celle qui maximise la concordance (paramètre **estimé**, pas choisi).

# %%
reconstructed = linked.groupby("lease_id").monthly_equivalent.sum()
reconstruction = leases[["lease_id", "lease_year", "rent_contract", "rent_effective", "has_concession"]].copy()
reconstruction["reconstructed_discount"] = reconstruction.lease_id.map(reconstructed).fillna(0.0)
reconstruction["actual_discount"] = reconstruction.rent_contract - reconstruction.rent_effective
reconstruction["error"] = reconstruction.reconstructed_discount - reconstruction.actual_discount
has_promo = set(linked[linked.charge_code == "PromoPay"].lease_id)
reconstruction["has_promopay"] = reconstruction.lease_id.isin(has_promo)

close = reconstruction.error.abs() <= CONFIG.exact_tolerance_dollars
close_label = f"reconstruits à ≤ {fmt(CONFIG.exact_tolerance_dollars, 0)} $"
summary_reconstruction = pd.DataFrame({
    "baux": reconstruction.groupby("has_promopay").size(),
    close_label: close.groupby(reconstruction.has_promopay).sum(),
    "part (%)": close.groupby(reconstruction.has_promopay).mean() * 100}).rename(index={False: "sans PromoPay", True: "avec PromoPay"})
summary_reconstruction.loc["tous"] = [len(reconstruction), close.sum(), close.mean() * 100]
print(summary_reconstruction.round(CONFIG.prose_digits).to_string())
print(f"\nConcordance par année de début de bail (part des baux {close_label}) :")
print((close.groupby(reconstruction.lease_year).mean() * 100).round(CONFIG.prose_digits).to_dict())


def reconstruction_share(grace_days):
    attached = link_concession_lines(concessions, leases, grace_days)
    discount = attached.groupby("lease_id").monthly_equivalent.sum().reindex(leases.lease_id).fillna(0.0)
    return float((np.abs(discount.to_numpy() - (leases.rent_contract - leases.rent_effective).to_numpy()) <= CONFIG.exact_tolerance_dollars).mean())


grace_test = pd.Series({days: reconstruction_share(days) * 100 for days in CONFIG.concession_grace_days_tested}, name=f"part des baux {close_label} (%)")
grace_test.index.name = "tolérance de rattachement (jours)"
print(grace_test.round(CONFIG.prose_digits).to_string())
assert grace_test.idxmax() == CONFIG.concession_link_grace_days, "la tolérance retenue doit maximiser la concordance"

# %%
promo_failures = (~close & reconstruction.has_promopay).sum() / (~close).sum()
narrate(f"**Lecture.** **{fmt(close.mean() * 100)} % des baux se reconstruisent à moins de {fmt(CONFIG.exact_tolerance_dollars, 0)} $** à partir des lignes, "
        "ce qui confirme (1) ce que mesure `rent_effective` (loyer contractuel moins les concessions étalées sur le bail) et (2) la lecture « forfait unique » de `PromoPay`. "
        f"Les écarts restants concernent des baux avec `PromoPay` dans {fmt(promo_failures * 100, 0)} % des cas (le moment exact du forfait dans le bail n'est pas toujours "
        f"étalé comme nous le supposons). **Décision :** nous utilisons `rent_effective` (champ calculé par Yardi, validé à {fmt(close.mean() * 100, 0)} % par notre "
        "reconstruction) pour toutes les analyses effectives, et nous gardons la reconstruction comme preuve de compréhension.")

# %% [markdown]
# **Que se passe-t-il si l'on prend la description au pied de la lettre ?** Un `PromoPay` mensuel pendant toute la durée du bail ferait payer presque rien aux locataires : la lecture littérale est absurde, ce que le test suivant chiffre (**étape qui crée** la table `promo_lines`, les seules lignes `PromoPay`).

# %%
promo_lines = linked[linked.charge_code == "PromoPay"]
literal_monthly_discount = promo_lines.groupby("lease_id").apply(lambda g: (-g.amount).sum(), include_groups=False)    # lecture littérale : le montant s'applique chaque mois
literal = leases.set_index("lease_id").loc[literal_monthly_discount.index]
literal_share_of_rent = literal_monthly_discount / literal.rent_contract
amortised_share = promo_lines.assign(share=lambda d: d.monthly_equivalent / d.rent_contract).groupby("lease_id").share.sum()
print(f"Baux avec PromoPay : {len(promo_lines.lease_id.unique()):,}")
print(f"  lecture littérale (le montant s'applique CHAQUE mois) : rabais médian = {literal_share_of_rent.median():.0%} du loyer mensuel (≈ loyer gratuit)")
print(f"  lecture forfait unique étalé sur le bail             : rabais médian = {amortised_share.median():.1%} du loyer mensuel")
assert close.mean() > CONFIG.min_reconstruction_share, f"la reconstruction doit confirmer rent_effective sur au moins {CONFIG.min_reconstruction_share:.0%} des baux"

# %% [markdown]
# ### 5c. Pourquoi contractuel et effectif divergent à partir de 2024
# Trois mesures annuelles : la **pénétration** (part des baux avec concession), la **profondeur** (rabais moyen parmi les baux qui en ont) et le **drag** (rabais moyen sur tous les baux) `= pénétration × profondeur`.

# %%
def concession_profile_by(frame, by):
    grouped = frame.groupby(by)
    return pd.DataFrame({
        "leases": grouped.size(),
        "penetration_pct": grouped.has_concession.mean() * 100,
        "depth_pct": grouped.apply(lambda g: g.loc[g.drag > 0, "drag"].mean() * 100, include_groups=False),
        "drag_pct": grouped.drag.mean() * 100})


profile_by_year = concession_profile_by(leases, "lease_year")
display(profile_by_year.round(CONFIG.table_digits))
profile_by_year_type = concession_profile_by(leases[leases.lease_year >= CONFIG.extended_backtest_years[0]], ["lease_year", "is_renewal"])
display(profile_by_year_type.round(CONFIG.table_digits))
recent_leases = leases[leases.lease_year >= CONFIG.backtest_years[0]]
quarterly_drag = recent_leases.groupby(recent_leases.lease_start.dt.to_period("Q")).agg(leases=("drag", "size"), drag_pct=("drag", lambda s: s.mean() * 100))
display(quarterly_drag.round(CONFIG.table_digits).T)

# %%
quarter_year = quarterly_drag.index.year
within_year_spread = quarterly_drag.drag_pct.groupby(quarter_year).std().mean()
between_year_spread = quarterly_drag.drag_pct.groupby(quarter_year).mean().std()
renewal_drag = profile_by_year_type.drag_pct.unstack("is_renewal")
recent_years = list(CONFIG.backtest_years)
narrate(f"**Lecture.** Le drag est une **fonction en escalier de l'année** : "
        f"{' ; '.join(f'{year} : {fmt(profile_by_year.drag_pct[year])} %' for year in recent_years)}. "
        f"Dans une même année il est stable d'un trimestre à l'autre (écart-type moyen entre trimestres : {fmt(within_year_spread, CONFIG.table_digits)} point, "
        f"contre {fmt(between_year_spread, CONFIG.table_digits)} entre années) : c'est une **politique annuelle** de concessions. "
        f"Il augmente à la fois par la pénétration ({' → '.join(fmt(profile_by_year.penetration_pct[year], 0) + ' %' for year in recent_years)}) "
        f"et par la profondeur ({' → '.join(fmt(profile_by_year.depth_pct[year]) + ' %' for year in recent_years)}). "
        f"Les renouvellements, d'abord moins concernés ({fmt(renewal_drag.loc[recent_years[0], 1])} % contre {fmt(renewal_drag.loc[recent_years[0], 0])} % en {recent_years[0]}), "
        f"rejoignent les relocations en {recent_years[-1]} ({fmt(renewal_drag.loc[recent_years[-1], 1])} % contre {fmt(renewal_drag.loc[recent_years[-1], 0])} %).")

# %% [markdown]
# **Décomposition exacte de l'écart contractuel − effectif.** Pour une paire, `(1 + g_eff) = (1 + g_contrat) × ((1 − drag) / (1 − drag_précédent)) ^ (1/gap)` : l'écart de croissance vaut à peu près le **changement du drag**. On décompose ce changement en part due à la pénétration et part due à la profondeur (moyennes de point milieu, la somme est exacte).

# %%
drag_change = profile_by_year[["penetration_pct", "depth_pct", "drag_pct"]].copy() / 100
changes = drag_change.diff()
midpoints = (drag_change + drag_change.shift(1)) / 2
divergence = pd.DataFrame({
    "contractuel (médiane)": same_unit[("contractuel", "median")],
    "effectif (médiane)": same_unit[("effectif", "median")]})
divergence["écart (points)"] = divergence.iloc[:, 0] - divergence.iloc[:, 1]
divergence["Δ drag (points)"] = changes.drag_pct * 100
divergence["… dû à la pénétration"] = changes.penetration_pct * midpoints.depth_pct * 100
divergence["… dû à la profondeur"] = midpoints.penetration_pct * changes.depth_pct * 100
divergence = divergence.loc[CONFIG.first_year + 1:CONFIG.last_observed_year]
assert np.allclose(divergence["… dû à la pénétration"] + divergence["… dû à la profondeur"], divergence["Δ drag (points)"]), "la décomposition doit se refermer"
divergence.round(CONFIG.table_digits)

# %%
pre_divergence = divergence.loc[:CONFIG.backtest_years[0], "écart (points)"].abs().max()
year_mid, year_last = CONFIG.backtest_years[1], CONFIG.backtest_years[2]
narrate(f"**Lecture.** Jusqu'en {CONFIG.backtest_years[0]}, les deux croissances sont à au plus {fmt(pre_divergence)} point l'une de l'autre. "
        f"Ensuite l'écart suit le **changement du drag** : en {year_mid}, le drag gagne {fmt(divergence.loc[year_mid, 'Δ drag (points)'])} points "
        f"(dont {fmt(divergence.loc[year_mid, '… dû à la pénétration'])} par la pénétration) ; en {year_last}, {fmt(divergence.loc[year_last, 'Δ drag (points)'])} points "
        f"({fmt(divergence.loc[year_last, '… dû à la pénétration'])} par la pénétration, {fmt(divergence.loc[year_last, '… dû à la profondeur'])} par la profondeur), "
        f"et l'écart contractuel − effectif vaut alors {fmt(divergence.loc[year_last, 'écart (points)'])} points. L'écart ne reproduit pas exactement le Δ drag "
        "parce que la médiane des paires n'est pas la moyenne des baux : l'identité est exacte pour chaque paire, approximative pour une médiane. "
        "**Les propriétaires ont relevé les loyers inscrits au bail et ont redonné une partie de la différence en concessions.**\n\n"
        "**Lequel citer à un propriétaire ?** Le **loyer effectif** pour budgéter les revenus (il additionne ce qui est réellement encaissé) ; le **loyer contractuel** "
        "pour le TAL et pour les lettres de renouvellement (c'est lui qui est encadré). Nous retenons l'effectif comme définition principale et publions les deux.")

# %%
fig, axes = plt.subplots(1, 2, figsize=STYLE.figsize_wide)
years_profile = profile_by_year.index
axes[0].plot(years_profile, profile_by_year.penetration_pct, "o-", color=PALETTE["contract"], label="pénétration (% des baux)")
axes[0].plot(years_profile, profile_by_year.depth_pct, "o-", color=PALETTE["effective"], label="profondeur (% du loyer)")
axes[0].plot(years_profile, profile_by_year.drag_pct, "o--", color="black", label="drag moyen (% du loyer)")
axes[0].set(title="Concessions : de plus en plus fréquentes et profondes", xlabel="année de début de bail")
axes[0].legend(fontsize=STYLE.legend_fontsize)
years_gap = divergence.index
axes[1].plot(years_gap, divergence["contractuel (médiane)"], "o-", color=PALETTE["contract"], label="contractuel")
axes[1].plot(years_gap, divergence["effectif (médiane)"], "o-", color=PALETTE["effective"], label="effectif")
axes[1].fill_between(years_gap, divergence["effectif (médiane)"], divergence["contractuel (médiane)"], color=PALETTE["effective"], alpha=STYLE.band_alpha, label="écart = concessions")
axes[1].set(title="Croissance à unité constante (médiane des paires, %)", xlabel="année de début de bail")
axes[1].legend(fontsize=STYLE.legend_fontsize)
plt.tight_layout()
plt.show()
