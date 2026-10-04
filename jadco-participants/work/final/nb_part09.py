# %% [markdown]
# ## 10. Prévoir la hausse 2026 et la valider par backtest
#
# ### 10a. Principe : un modèle structurel par composantes, sans fuite d'information
# La section 8 a montré qu'aucune série publique ne prévoit bien la hausse *globale* ; en revanche chaque **composante** a un moteur distinct. Le modèle retenu prévoit chacune avec une règle **sans paramètre ajusté** ou tirée du droit :
#
# | Composante | Moteur | Règle du modèle structurel |
# |---|---|---|
# | **Renouvellements** (contractuel) | politique de la société, encadrée par le TAL | politique commune ancrée sur le TAL ; calendrier des avis du Québec utilisé comme proxy de politique, y compris pour The Met exempté (section 10d) |
# | **Relocations** (contractuel) | marché | variation de l'IPC loyers du Québec de l'année précédente (section 8b, 10c) |
# | **Concessions** (niveau du drag des nouveaux baux) | politique de prix | variation de l'IPC loyers du Québec de l'année précédente (section 10c), avec deux trajectoires alternatives quand l'IPC baisse (section 10d) |
#
# **Distribution plutôt que moyenne.** Les centres alimentent une **simulation Monte-Carlo de la cohorte réelle** : les unités dont le bail échoit dans l'année `t` et qui sont connues à la date de coupure, leur loyer et leur concession actuels, leur probabilité de renouveler (par province, sur les dernières années). Chaque unité tire son type, sa dispersion autour du centre (résidus de l'an dernier) et sa nouvelle concession **dans la distribution de sa propre propriété** (The Met et les immeubles en location initiale concèdent plus) ; la médiane des paires simulées est la prévision. C'est la **même définition** que la cible, ce qui évite de confondre « médiane d'un mélange » et « mélange de médianes ».
#
# **Comparateur générique.** Pour mesurer ce qu'apporte la structure, le backtest évalue aussi une **combinaison de règles candidates pondérée par l'inverse de leur erreur passée** (reconduction, moyenne, TAL + écart, IPC + écart, affichages, tendances des concessions…), où les poids ne voient que les années antérieures à l'année prédite. C'est ce qu'obtiendrait une méthode qui ne connaît pas la structure.
#
# **Trois millésimes d'information**, rapportés séparément (dates dans `ForecastConfig.vintage_dates`) :
# * `december` : tout ce qui est public au 31 décembre de `t−1` (le TAL de l'année `t` n'est pas encore connu : il est prévu par sa formule, section 10c) ;
# * `january` : ajoute le TAL de l'année `t` (publié vers le 20 janvier) ;
# * `october` : ajoute l'IPC des premiers mois de l'année `t`. **C'est le millésime de la prévision en direct** (données publiques à la date de retrait) ; il se rejoue de la même façon dans le passé.
#
# Dans tous les millésimes, les **baux** sont coupés au 31 décembre de `t−1` : seule l'information publique change d'un millésime à l'autre (choix prudent).

# %%
# CONFIG (prévision)
@dataclass(frozen=True)
class ForecastConfig:
    vintage_dates: dict = field(default_factory=lambda: {"december": (-1, "12-31"), "january": (0, "01-31"), "october": (0, CONFIG.data_as_of[5:])})  # (décalage d'année, mois-jour) ; october = date de retrait
    live_vintage: str = "october"    # prévision en direct : information publique à la date de retrait
    cpi_min_months: int = 3          # pas de moyenne IPC avec moins de trois mois publiés
    n_simulations_backtest: int = 1000
    renewal_share_window_years: int = 2     # probabilité de renouveler : moyenne des deux dernières années (part stable, section 6c)
    min_cell_multiplier: int = 2            # une province exige deux fois la taille minimale de cellule par année de fenêtre
    penetration_k_window_years: int = 2     # sensibilité k de la pénétration des concessions : moyenne des deux dernières années connues (section 10c ; testé en 10f bis)
    drag_pools: str = "property"            # forme des concessions : distribution de la propriété de l'unité ("property") ou du portefeuille ("portfolio") ; testé en 10f bis
    notice_timing: str = "uniform"          # date de l'avis dans la fenêtre légale : "uniform" (retenu), "earliest" ou "latest" (bornes, testées en 10f bis)
    robustness_variants: dict = field(default_factory=lambda: {
        "k : dernière année seulement": {"penetration_k_window_years": 1}, "k : trois dernières années": {"penetration_k_window_years": 3},
        "k : tout l'historique": {"penetration_k_window_years": None},
        "part de renouvellement : dernière année": {"renewal_share_window_years": 1}, "part de renouvellement : trois ans": {"renewal_share_window_years": 3},
        "concessions : distribution du portefeuille": {"drag_pools": "portfolio"},
        "avis : au plus tôt de la fenêtre légale": {"notice_timing": "earliest"}, "avis : au plus tard de la fenêtre légale": {"notice_timing": "latest"}})
    mc_precision_seeds: int = 5             # graines indépendantes pour mesurer la précision Monte-Carlo du headline (section 11c)
    # --- TAL : faits juridiques (sources en section 13) ---
    tal_new_format_first_year: int = 2026   # nouvelle méthode (pourcentage de base) pour les avis donnés à compter du 1er janvier 2026 (décret 1455-2025)
    tal_new_method_window_years: int = 3    # décret 1455-2025, art. `3.1` : pourcentage de base = moyenne des trois variations annuelles de la moyenne sur 12 mois de l'IPC d'ensemble du Québec
    notice_months_long_lease: tuple = (3, 6)   # art. 1942 C.c.Q. : avis de modification 3 à 6 mois avant la fin d'un bail de 12 mois ou plus
    notice_months_short_lease: tuple = (1, 2)  # ... et 1 à 2 mois avant la fin d'un bail de moins de 12 mois
    max_tal_rebuild_error: float = 0.1      # écart maximal toléré entre la formule du TAL reconstruite et le taux publié (points)
    min_indexation_correlation: float = 0.9 # corrélation minimale exigée entre le drag annuel et l'IPC loyers de l'année précédente
    # --- comparateur générique (combinaison de règles) ---
    shrink_k: float = 2.0            # l'écart moyen « interne − règle » est rétréci vers 0 : somme / (n + k), comme k années fictives à écart nul
    min_scored_years: int = 2        # une règle n'a un poids propre qu'après deux années d'erreurs observées
    error_decay: float = 0.5         # une année plus ancienne pèse moitié moins dans l'erreur moyenne d'une règle
    damping: float = 0.5             # part du dernier changement conservée par la règle « tendance amortie »


FORECAST = ForecastConfig()
EXTERNAL_TABLES = {"annual": external, "cpi": cpi_monthly}


@contextmanager
def forecast_settings(**changes):
    """Remplace temporairement des réglages de ForecastConfig (tests de robustesse), puis rétablit les réglages retenus."""
    global FORECAST
    retained = FORECAST
    FORECAST = replace(retained, **changes)
    try:
        yield
    finally:
        FORECAST = retained


def vintage_cutoff(year, vintage):
    """Date jusqu'à laquelle une valeur publique est connue pour prévoir `year` selon le millésime."""
    year_offset, month_day = FORECAST.vintage_dates[vintage]
    return pd.Timestamp(f"{year + year_offset}-{month_day}")


def known_at(leases_table, asking_table, year):
    """Baux et affichages connus à la fin de year − 1 (coupure appliquée AVANT tout calcul)."""
    cutoff = year_end(year - 1)
    return leases_table[leases_table.lease_start <= cutoff], asking_table[asking_table.ask_date <= cutoff]


def cpi_driver(external_tables, year, cutoff):
    """Moyenne des variations annuelles de l'IPC loyers du Québec, mois de `year` publiés avant `cutoff`."""
    monthly = external_tables["cpi"]
    monthly = monthly[(monthly.geo == "Quebec") & (monthly.published_on <= cutoff) & (monthly.month.dt.year == year)].yoy_pct.dropna()
    return float(monthly.mean()) if len(monthly) >= FORECAST.cpi_min_months else np.nan


def published_series(external_tables, series, cutoff):
    rows = external_tables["annual"]
    return rows[(rows.series == series) & (rows.published_on <= cutoff)].drop_duplicates("year", keep="last").set_index("year").value.sort_index()


def tal_slope(external_tables, cutoff):
    """Pente de « estimation ancienne méthode ≈ pente × IPC loyers QC(t−1) », estimée par MCO passant par l'origine sur les seuls taux publiés à `cutoff`."""
    old = published_series(external_tables, "tal_recommended_unheated_old_method", cutoff)
    previous_cpi = pd.Series({year: cpi_driver(external_tables, year - 1, cutoff) for year in old.index}).dropna()
    common = old.index.intersection(previous_cpi.index)
    if len(common) < FORECAST.min_scored_years:
        return np.nan
    return float((old[common] * previous_cpi[common]).sum() / (previous_cpi[common] ** 2).sum())


def tal_old_method_rate(external_tables, year, cutoff):
    """Taux « ancienne méthode » de `year` : publié s'il l'est à `cutoff`, sinon prévu par sa formule (pente × IPC de year − 1), sinon le dernier publié."""
    old = published_series(external_tables, "tal_recommended_unheated_old_method", cutoff)
    if year in old.index:
        return float(old[year])
    formula = tal_slope(external_tables, cutoff) * cpi_driver(external_tables, year - 1, cutoff)
    if pd.notna(formula):
        return float(formula)
    return float(old.iloc[-1]) if len(old) else np.nan


def tal_base_percentage(external_tables, year, cutoff):
    """Pourcentage de base (nouvelle méthode) de `year` : publié s'il l'est à `cutoff`, sinon sa formule (moyenne de trois ans de l'IPC d'ensemble
    du Québec publiée à `cutoff`), sinon NaN."""
    new = published_series(external_tables, "tal_base_percentage_new_format", cutoff)
    if year in new.index:
        return float(new[year])
    all_items_known = published_series(external_tables, "cpi_all_items_avg_yoy", cutoff).loc[year - FORECAST.tal_new_method_window_years:year - 1]
    return max(float(all_items_known.mean()), 0.0) if len(all_items_known) == FORECAST.tal_new_method_window_years else np.nan     # un résultat négatif est ramené à zéro


def prepare_context(known_leases, known_asking):
    """Tout ce qui se calcule à partir des seules données connues à la date de coupure."""
    known_leases = known_leases.sort_values(UNIT_KEY + ["lease_start"]).reset_index(drop=True)
    known_pairs = build_pairs(known_leases)
    known_pairs = known_pairs[known_pairs.is_valid].copy()
    quebec_pairs = known_pairs[known_pairs.province == "Quebec"]
    annual_series = pd.DataFrame({
        "renew_qc": quebec_pairs[quebec_pairs.is_renewal == 1].groupby("lease_year").growth_contract_pct.median(),
        "reloc_qc": quebec_pairs[quebec_pairs.is_renewal == 0].groupby("lease_year").growth_contract_pct.median(),
        "all_contract": known_pairs.groupby("lease_year").growth_contract_pct.median(),
        "all_effective": known_pairs.groupby("lease_year").growth_effective_pct.median(),
        "drag": known_leases.groupby("lease_year").drag.mean() * 100,
        "penetration": known_leases.groupby("lease_year").has_concession.mean() * 100,
        "depth": known_leases[known_leases.drag > 0].groupby("lease_year").drag.mean() * 100,
        "ask": same_unit_asking_growth(known_asking).groupby("ask_year").growth_pct.median() if len(known_asking) else pd.Series(dtype=float)})
    return {"leases": known_leases, "pairs": known_pairs, "annual": annual_series}


def shrunk_mean(values):
    series = pd.Series(list(values), dtype=float).dropna()
    return float(series.sum() / (len(series) + FORECAST.shrink_k)) if len(series) else 0.0


def rule_forecasts(context, external_tables, component, year, vintage):
    """Prévision de `component` pour `year` par chaque règle candidate du comparateur, avec la seule information antérieure à `year`."""
    cutoff = vintage_cutoff(year, vintage)
    annual_series = context["annual"]
    history = annual_series[component].loc[:year - 1].dropna()
    rules = {"reconduire": history.iloc[-1] if len(history) else np.nan, "moyenne historique": history.mean() if len(history) else np.nan}
    if component == "renew_qc":
        spreads = [annual_series.loc[y, component] - tal_old_method_rate(external_tables, y, vintage_cutoff(y, vintage)) for y in history.index if y >= CONFIG.extended_backtest_years[0]]
        rules["TAL + écart"] = tal_old_method_rate(external_tables, year, cutoff) + shrunk_mean(spreads)
        rules["TAL seul"] = tal_old_method_rate(external_tables, year, cutoff)
    if component in ("renew_qc", "reloc_qc"):
        spreads = [annual_series.loc[y, component] - cpi_driver(external_tables, y - 1, vintage_cutoff(y, vintage)) for y in history.index]
        rules["IPC décalé + écart"] = cpi_driver(external_tables, year - 1, cutoff) + shrunk_mean(spreads)
    if component == "reloc_qc":
        gap_history = (annual_series.reloc_qc - annual_series.renew_qc).loc[:year - 1].dropna()
        renewal_years = [y for y in gap_history.index if y >= CONFIG.extended_backtest_years[0]]
        if len(gap_history):
            rules["TAL + écart relocation−renouvellement"] = tal_old_method_rate(external_tables, year, cutoff) + float(gap_history.iloc[-1]) + shrunk_mean(
                [annual_series.loc[y, "renew_qc"] - tal_old_method_rate(external_tables, y, vintage_cutoff(y, vintage)) for y in renewal_years])
        ask_history = annual_series["ask"].dropna()
        spreads = [annual_series.loc[y, component] - ask_history[y - 1] for y in history.index if (y - 1) in ask_history.index]
        rules["affichés décalés + écart"] = (ask_history.get(year - 1, np.nan) + shrunk_mean(spreads)) if (year - 1) in ask_history.index else np.nan
        if vintage == "october":
            spreads = [annual_series.loc[y, component] - cpi_driver(external_tables, y, vintage_cutoff(y, vintage)) for y in history.index]
            rules["IPC de l'année (nowcast) + écart"] = cpi_driver(external_tables, year, cutoff) + shrunk_mean(spreads)
    if component == "drag":
        deltas = history.diff().dropna()
        last_delta = deltas.iloc[-1] if len(deltas) else 0.0
        rules["tendance"] = history.iloc[-1] + last_delta
        rules["tendance moyenne"] = history.iloc[-1] + (deltas.tail(CONFIG.recent_years_for_mean).mean() if len(deltas) else 0.0)
        rules["tendance amortie"] = history.iloc[-1] + FORECAST.damping * last_delta
        penetration, depth = annual_series["penetration"].loc[:year - 1].dropna(), annual_series["depth"].loc[:year - 1].dropna()
        if len(penetration) > 1 and len(depth) > 1:
            penetration_forecast = min(100, penetration.iloc[-1] + penetration.diff().iloc[-1])     # la pénétration ne dépasse pas 100 %
            rules["pénétration × profondeur (tendances)"] = penetration_forecast * (depth.iloc[-1] + depth.diff().iloc[-1]) / 100
        rules.pop("moyenne historique")
    return rules


def decayed_mean_error(errors):
    weights = FORECAST.error_decay ** np.arange(len(errors))[::-1]
    return float(np.sum(weights * np.asarray(errors)) / weights.sum())


def ensemble_forecast(context, external_tables, component, year, vintage):
    """Combinaison des règles, pondérée par l'inverse de leur erreur passée (récente d'abord). Retourne (prévision, poids, prévisions)."""
    forecasts = {name: value for name, value in rule_forecasts(context, external_tables, component, year, vintage).items() if pd.notna(value)}
    past_errors = {}
    for past_year in context["annual"][component].loc[CONFIG.first_year + 1:year - 1].dropna().index:
        for name, value in rule_forecasts(context, external_tables, component, past_year, vintage).items():
            if name in forecasts and pd.notna(value):
                past_errors.setdefault(name, []).append(abs(value - context["annual"].loc[past_year, component]))
    scored = {name: decayed_mean_error(errors) for name, errors in past_errors.items() if len(errors) >= FORECAST.min_scored_years}
    if len(scored) >= 2:
        inverse = {name: 1 / max(error, np.finfo(float).eps) for name, error in scored.items()}
        weights = {name: value / sum(inverse.values()) for name, value in inverse.items()}
    else:
        weights = {name: 1 / len(forecasts) for name in forecasts}
    center = sum(weights[name] * forecasts[name] for name in weights)
    return center, weights, forecasts

# %% [markdown]
# ### 10b. Cohorte et simulation
# La **cohorte de l'année `t`** = les unités dont le dernier bail connu se termine de sorte que le suivant commence pendant `t` (le nouveau bail commence le lendemain de la fin de l'ancien). On connaît leur loyer contractuel, leur drag actuel et leur durée, donc la **base de comparaison** de chacune des paires est déjà fixée ; seul le nouveau bail reste à prévoir.
#
# **Concessions des nouveaux baux : deux marges.** Le niveau moyen du drag `D` est une entrée (section 10c). Sa *forme* compte autant que son niveau pour une médiane : en 2025 la pénétration saute, et une simple mise à l'échelle de la distribution de l'an dernier garderait trop de baux sans concession. On sépare donc :
# * la **marge extensive** (qui reçoit une concession) : `pénétration = 1 − exp(−k × D)`, la relation de la section 10c, avec `k` **estimé à la date de coupure** sur les `penetration_k_window_years` dernières années connues ;
# * la **marge intensive** (combien) : la profondeur moyenne vaut `D / pénétration` ; chaque unité tire son rabais dans la distribution des rabais positifs de l'an dernier **de sa propre propriété** (si elle compte au moins `min_cell_size` baux, sinon celle du portefeuille), mise à l'échelle de cette profondeur, jamais au-delà du plus grand rabais observé (support empirique, pas un plafond choisi).

# %%
def build_cohort(context, year):
    latest = context["leases"].sort_values("lease_start").groupby(UNIT_KEY).tail(1)
    starts_next = latest.lease_end + pd.Timedelta(days=1)          # le nouveau bail commence le lendemain de la fin de l'ancien
    return latest[starts_next.dt.year == year].assign(next_start=starts_next).reset_index(drop=True)


def renewal_probability(context, year, cohort):
    recent = context["pairs"][context["pairs"].lease_year.between(year - FORECAST.renewal_share_window_years, year - 1)]
    if recent.empty:
        raise ValueError("Aucune paire récente pour estimer la probabilité de renouvellement")
    pooled = recent.is_renewal.mean()
    by_province = recent.groupby("province").is_renewal.agg(["mean", "size"])
    reliable = by_province["size"] >= CONFIG.min_cell_size * FORECAST.renewal_share_window_years * FORECAST.min_cell_multiplier
    return cohort.province.map(lambda province: by_province.loc[province, "mean"] if province in by_province.index and reliable[province] else pooled).to_numpy()


def penetration_sensitivity(context, year):
    """k de « pénétration = 1 − exp(−k × drag) », moyenne des dernières années connues avant `year` (estimé à la date de coupure)."""
    annual_series = context["annual"].loc[:year - 1].dropna(subset=["penetration", "drag"])
    annual_series = annual_series[(annual_series.drag > 0) & annual_series.penetration.between(0, 100, inclusive="neither")]
    if annual_series.empty:
        raise ValueError("Aucune année valide pour estimer la pénétration des concessions")
    k_by_year = -np.log1p(-annual_series.penetration / 100) / annual_series.drag
    window = FORECAST.penetration_k_window_years
    return float((k_by_year.tail(window) if window else k_by_year).mean())


def new_lease_drags(context, cohort, year, drag_level, n_draws, rng):
    """Concession de chaque nouveau bail (fraction du loyer) : marge extensive (pénétration) puis marge intensive (profondeur, par propriété)."""
    if not np.isfinite(drag_level) or drag_level < 0:
        raise ValueError("Le drag doit être fini et positif ou nul")
    if drag_level == 0:
        return np.zeros((n_draws, len(cohort)))
    penetration = -np.expm1(-penetration_sensitivity(context, year) * drag_level)
    last_year = context["leases"][(context["leases"].lease_year == year - 1) & (context["leases"].drag > 0)]
    portfolio_pool = last_year.drag.to_numpy()
    if not len(portfolio_pool):
        raise ValueError("Aucune concession positive l’année précédente : profondeur non identifiable")
    pools = {code: group.drag.to_numpy() for code, group in last_year.groupby("prop_code") if len(group) >= CONFIG.min_cell_size} if FORECAST.drag_pools == "property" else {}
    depths = np.empty((n_draws, len(cohort)))
    for code, columns in cohort.groupby("prop_code").indices.items():
        depths[:, columns] = rng.choice(pools.get(code, portfolio_pool), (n_draws, len(columns)))
    depths *= (drag_level / 100 / penetration) / portfolio_pool.mean()
    has_concession = rng.random((n_draws, len(cohort))) < penetration
    return np.clip(np.where(has_concession, depths, 0.0), 0, context["leases"].drag.max())


def simulate_cohort(context, cohort, year, renewal_center, relocation_center, drag_level, n_draws, seed, ontario_regime="exempt", ontario_guideline=None):
    """Monte-Carlo de la cohorte : paires simulées (contractuel et effectif annualisés, en %). `renewal_center` peut être propre à chaque unité."""
    if ontario_regime not in ("exempt", "capped"):
        raise ValueError("Le régime Ontario doit être exempt ou capped")
    if ontario_regime == "capped" and (ontario_guideline is None or not np.isfinite(ontario_guideline)):
        raise ValueError("Le régime capped exige une ligne directrice Ontario finie")
    if cohort.empty or n_draws <= 0:
        raise ValueError("La simulation exige une cohorte non vide et des tirages positifs")
    if not np.isfinite(renewal_center).all() or not np.isfinite(relocation_center):
        raise ValueError("Les centres de croissance doivent être finis")
    if not cohort.drag.between(0, 1, inclusive="left").all():
        raise ValueError("Les concessions précédentes doivent être comprises entre zéro et un exclu")
    rng = np.random.default_rng(seed)
    last_pairs = context["pairs"][context["pairs"].lease_year == year - 1]
    residuals = {flag: (last_pairs[last_pairs.is_renewal == flag].growth_contract_pct
                        - last_pairs[last_pairs.is_renewal == flag].growth_contract_pct.median()).to_numpy() for flag in (0, 1)}
    if any(not len(pool) for pool in residuals.values()):
        raise ValueError("Renouvellements et relocations exigent chacun des paires l’année précédente")
    n_units = len(cohort)
    renew = rng.random((n_draws, n_units)) < renewal_probability(context, year, cohort)
    growth_renewal = np.asarray(renewal_center, dtype=float) + rng.choice(residuals[1], (n_draws, n_units))
    growth_relocation = relocation_center + rng.choice(residuals[0], (n_draws, n_units))
    contract = np.where(renew, growth_renewal, growth_relocation)
    if ontario_regime == "capped" and ontario_guideline is not None:
        in_ontario = (cohort.province == "Ontario").to_numpy()
        contract = np.where(in_ontario & renew, np.minimum(contract, ontario_guideline), contract)
    new_drag = new_lease_drags(context, cohort, year, drag_level, n_draws, rng)
    previous_drag = cohort.drag.to_numpy()[None, :]
    gap_years = ((cohort.next_start - cohort.lease_start).dt.days.to_numpy() / CONFIG.days_per_year)[None, :]
    if not np.isfinite(gap_years).all() or (gap_years <= 0).any():
        raise ValueError("L’écart entre débuts de bail doit être fini et strictement positif")
    effective = ((1 + contract / 100) * ((1 - new_drag) / (1 - previous_drag)) ** (1 / gap_years) - 1) * 100
    return {"cohort": cohort, "renew": renew, "contract": contract, "effective": effective, "weights": cohort.rent_contract.to_numpy()}


def summarise_simulation(simulation, segment_keys=()):
    """Statistiques par tirage : médianes (définition de la cible), moyennes pondérées par le loyer, effet moyen des concessions, segments."""
    contract, effective, renew, weights = simulation["contract"], simulation["effective"], simulation["renew"], simulation["weights"]
    summary = {"contract": np.median(contract, axis=1), "effective": np.median(effective, axis=1),
               "renewal_contract": np.nanmedian(np.where(renew, contract, np.nan), axis=1),
               "relocation_contract": np.nanmedian(np.where(~renew, contract, np.nan), axis=1),
               "contract_mean": contract @ weights / weights.sum(), "effective_mean": effective @ weights / weights.sum(),
               "concession_effect": (effective - contract).mean(axis=1), "segments": {}}
    for keys in segment_keys:
        for label, columns in simulation["cohort"].groupby(list(keys)).indices.items():
            label = label if isinstance(label, tuple) else (label,)
            summary["segments"][(keys, label)] = {"contract": np.median(contract[:, columns], axis=1), "effective": np.median(effective[:, columns], axis=1),
                                                  "units": len(columns)}
    return summary

# %% [markdown]
# **Pourquoi une distribution de concessions par propriété ?** Parce que les propriétés ne concèdent pas la même chose : le tableau suivant donne, pour la dernière année observée, la pénétration, la profondeur et le drag de chaque propriété. Le choix est ensuite testé contre la distribution du portefeuille (section 10f bis).

# %%
property_concessions = concession_profile_by(leases[leases.lease_year == CONFIG.last_observed_year], "prop_code")
display(property_concessions.round(CONFIG.table_digits))
narrate(f"**Lecture.** En {CONFIG.last_observed_year}, le drag va de {fmt(property_concessions.drag_pct.min())} % (`{property_concessions.drag_pct.idxmin()}`) à "
        f"{fmt(property_concessions.drag_pct.max())} % (`{property_concessions.drag_pct.idxmax()}`), et la profondeur de {fmt(property_concessions.depth_pct.min())} à "
        f"{fmt(property_concessions.depth_pct.max())} % : un écart de {fmt(property_concessions.drag_pct.max() - property_concessions.drag_pct.min())} points entre propriétés, "
        "du même ordre que la variation annuelle du drag. Une unité de la cohorte tire donc sa concession dans la distribution de sa propre propriété.")

# %% [markdown]
# ### 10c. Une découverte : concessions, relocations et TAL suivent l'IPC loyers de l'année précédente
# La comparaison des séries historiques révèle trois relations simples. **Elles ont été repérées en regardant 2018-2025, donc le backtest qui suit les favorise** ; nous le disons, et nous montrons leur stabilité plutôt que de les supposer.
# 1. **Concessions.** Le drag moyen des baux d'une année `t` suit la variation de l'IPC loyers du Québec de `t−1`, **sans paramètre** : l'entreprise semble réinitialiser son budget de concessions chaque janvier sur l'inflation des loyers de l'année écoulée, ce qui explique les marches annuelles de la section 5.
# 2. **Relocations.** La hausse contractuelle des relocations du Québec suit la même variation de l'IPC de `t−1` (section 8b).
# 3. **TAL.** L'ancienne méthode du TAL vaut une fraction stable de cet IPC (pente estimée ci-dessous, sur les seuls taux publiés) ; la nouvelle méthode (décret 1455-2025, art. `3.1`) est la moyenne des trois variations annuelles de la moyenne sur 12 mois de l'IPC d'ensemble du Québec, non désaisonnalisé (résultat négatif ramené à zéro) : la formule doit retrouver les pourcentages publiés, ce que vérifie la cellule.

# %%
previous_year_cpi = pd.Series({year: cpi_driver(EXTERNAL_TABLES, year - 1, vintage_cutoff(year, "january")) for year in range(CONFIG.first_year - 1, CONFIG.forecast_year + 1)})
indexation = pd.DataFrame({"drag moyen des baux de l'année (%)": profile_by_year.drag_pct, "IPC loyers QC, année précédente (%)": previous_year_cpi,
                           "relocations QC (contractuel, %)": internal_series["QC relocations (contractuel)"], "TAL ancienne méthode (%)": TAL_OLD}).loc[CONFIG.first_year - 1:CONFIG.last_observed_year]
display(indexation.round(CONFIG.table_digits))
drag_cpi_correlation = indexation.iloc[:, 0].corr(indexation.iloc[:, 1])
drag_cpi_recent_error = (indexation.iloc[:, 0] - indexation.iloc[:, 1]).abs().loc[CONFIG.backtest_years[0]:].mean()
relocation_cpi_rmse = float(np.sqrt(((indexation.iloc[:, 2] - indexation.iloc[:, 1]) ** 2).mean()))
relocation_cpi_bias = float((indexation.iloc[:, 2] - indexation.iloc[:, 1]).mean())
live_cutoff = vintage_cutoff(CONFIG.forecast_year, FORECAST.live_vintage)
tal_slope_live = tal_slope(EXTERNAL_TABLES, live_cutoff)
tal_vs_cpi = indexation[["TAL ancienne méthode (%)", "IPC loyers QC, année précédente (%)"]].dropna()
tal_leave_one_out = [abs(tal_vs_cpi.iloc[position, 0] - float((tal_vs_cpi.iloc[:, 0].drop(year) * tal_vs_cpi.iloc[:, 1].drop(year)).sum()
                                                               / (tal_vs_cpi.iloc[:, 1].drop(year) ** 2).sum()) * tal_vs_cpi.iloc[position, 1])
                     for position, year in enumerate(tal_vs_cpi.index)]
all_items = external_series("cpi_all_items_avg_yoy", "Quebec")
tal_new_rebuilt = pd.Series({year: max(all_items.loc[year - FORECAST.tal_new_method_window_years:year - 1].mean(), 0.0) for year in TAL_NEW_FORMAT.index})
penetration_constant = (-np.log(1 - profile_by_year.penetration_pct / 100) / profile_by_year.drag_pct).loc[CONFIG.first_year - 1:]
print(f"TAL (ancienne méthode) ≈ {tal_slope_live:.3f} × IPC loyers QC de l'année précédente ; erreurs « un an de côté » : {[round(e, CONFIG.table_digits) for e in tal_leave_one_out]}")
print("Nouvelle méthode du TAL reconstruite :", tal_new_rebuilt.round(CONFIG.table_digits).to_dict(), "| publiée :", TAL_NEW_FORMAT.to_dict())
print("Pénétration des concessions : k = −ln(1 − pénétration) / drag =", penetration_constant.round(CONFIG.fine_digits).to_dict())
assert (tal_new_rebuilt - TAL_NEW_FORMAT).abs().max() < FORECAST.max_tal_rebuild_error, "la formule doit retrouver les taux publiés"
assert drag_cpi_correlation > FORECAST.min_indexation_correlation, "le drag doit suivre l'IPC loyers de l'année précédente"

# %%
narrate(f"**Lecture et prudence.** Le drag suit l'IPC loyers de l'année précédente avec une corrélation de **{fmt(drag_cpi_correlation, CONFIG.fine_digits)}** "
        f"et un écart absolu moyen de {fmt(drag_cpi_recent_error, CONFIG.table_digits)} point sur {CONFIG.backtest_years[0]}-{CONFIG.backtest_years[-1]}. "
        f"Les relocations s'en écartent de {fmt(relocation_cpi_rmse, CONFIG.table_digits)} point en écart quadratique moyen (biais moyen {fmt(relocation_cpi_bias, CONFIG.table_digits, signed=True)}) "
        f"sur {indexation.iloc[:, 2].notna().sum()} ans, sans paramètre ajusté. L'ancienne méthode du TAL vaut {fmt(tal_slope_live, CONFIG.fine_digits)} × cet IPC "
        f"(erreurs « un an de côté » de {fmt(min(tal_leave_one_out), CONFIG.table_digits)} à {fmt(max(tal_leave_one_out), CONFIG.table_digits)} point) et la nouvelle méthode "
        f"retrouve les pourcentages publiés à {fmt((tal_new_rebuilt - TAL_NEW_FORMAT).abs().max(), CONFIG.table_digits)} point près. "
        f"Enfin, la **pénétration** suit `1 − exp(−k × drag)` : `k` baisse de {fmt(penetration_constant.iloc[0], CONFIG.table_digits)} à "
        f"{fmt(penetration_constant.loc[CONFIG.backtest_years[0]], CONFIG.table_digits)} jusqu'en {CONFIG.backtest_years[0]}, puis se stabilise "
        f"({', '.join(fmt(penetration_constant[year], CONFIG.fine_digits) for year in CONFIG.backtest_years)}) ; le modèle l'estime donc sur les dernières années connues, à la date de coupure.\n\n"
        f"Huit points annuels ne prouvent pas une loi : ce sont des régularités **observées**, pas des règles écrites. Elles donnent deux lectures du TAL {CONFIG.forecast_year} : "
        f"le **pourcentage de base** de {fmt(TAL_NEW_FORMAT[CONFIG.forecast_year])} % (nouvelle méthode) et son **équivalent ancienne méthode** "
        f"({fmt(tal_slope_live * previous_year_cpi[CONFIG.forecast_year], CONFIG.table_digits)} %), à comparer à l'historique des renouvellements. La section suivante dit lequel s'applique à quelle unité.")

# %% [markdown]
# #### Précédents de baisse du moteur des concessions
# On compare l’IPC disponible au drag précédent, puis au drag effectivement observé. Les trajectoires diffèrent dès que le premier est inférieur au second ; il ne faut pas confondre absence de précédent récent et absence de précédent historique.

# %%
concession_precedents = indexation.copy()
concession_precedents["drag précédent (%)"] = profile_by_year.drag_pct.shift(1)
concession_precedents = concession_precedents[concession_precedents.iloc[:, 1] < concession_precedents["drag précédent (%)"]]
display(concession_precedents.round(CONFIG.table_digits))
narrate(f"**Lecture.** {len(concession_precedents)} année(s) de l’historique ont un IPC inférieur au drag précédent : "
        f"{', '.join(map(str, concession_precedents.index))}. Leurs concessions réalisées figurent dans le tableau ; "
        "ce constat empêche de présenter la baisse du moteur comme un événement sans précédent.")

# %% [markdown]
# ### 10d. 2026 : dater le changement de méthode du TAL, et ce que les données ne peuvent pas trancher
# **Ancrage daté par l'avis (fait juridique).** La nouvelle méthode ne s'applique qu'aux avis de modification donnés **à compter du 1er janvier 2026** (décret 1455-2025, art. 6 et 8) ; les avis antérieurs restent soumis à l'ancienne méthode. L'avis est donné dans une **fenêtre légale** avant l'arrivée du terme du bail (art. 1942 C.c.Q. : 3 à 6 mois pour un bail de 12 mois ou plus, 1 à 2 mois sinon). Pour chaque unité de la cohorte, on calcule donc, **à partir de sa propre date de fin de bail et de sa durée**, la part de sa fenêtre d'avis située avant le 1er janvier 2026 (avis supposé uniforme dans la fenêtre : la seule hypothèse) ; cette part reçoit le taux de l'ancienne méthode de la fenêtre de validité du bail (2025 pour un bail débutant au plus tard le 1er avril), le reste reçoit le pourcentage de base. Avant 2026, il n'y a pas de changement de méthode : l'ancrage est le taux de la fenêtre de validité.
#
# **The Met : hypothèse de politique, pas droit québécois.** Le calendrier des avis du Québec est aussi utilisé comme proxy de la politique commune de l’entreprise à Ottawa. Le droit québécois ne s’y applique pas. Sans dates d’avis de The Met, ce proxy reste une limite ; les deux bornes de calendrier sont testées en section 10f bis.
#
# **Ce qui est publié et ce qui est prévu.** Pour la fenêtre 2026, le TAL n'a pas publié d'estimation moyenne selon l'ancienne méthode ; le taux « ancienne méthode » 2026 est donc **prévu** par la formule de la section 10c (pente × IPC loyers de l'année précédente). Les composantes 2026 de l'ancienne méthode que le TAL publie (tableau 2 de sa page des pourcentages, avis donnés avant le 1er janvier 2026) indiquent un taux un peu plus bas : ce proxy penche légèrement vers le haut, ce qui ne touche que la part des avis donnés avant la réforme et la lecture « ancienne méthode pour tous ».
#
# **Deux choix de politique incertains**, insuffisamment identifiés par le backtest :
# 1. **La société suit-elle la loi au pied de la lettre ?** Lecture centrale : ancrage daté par l'avis. Lectures extrêmes : le pourcentage de base appliqué à tous les renouvellements de l'année, ou l'ancienne méthode appliquée à tous.
#    Le pourcentage du TAL n'est pas un plafond : c'est ce que le Tribunal appliquerait si le locataire refusait la hausse et que le propriétaire demandait la fixation. La société peut donc proposer plus (son habitude, l'ancienne méthode) ou s'aligner sur le nouveau pourcentage pour tous ; depuis 2023 ses renouvellements suivent le TAL (section 6b), mais jamais lors d'un changement de méthode.
# 2. **Que font les concessions quand l'IPC loyers baisse ?** C'est la première fois depuis le début du régime d'indexation : le drag a-t-il un **cliquet** (il ne baisse jamais), s'arrête-t-il **à mi-chemin** entre l'indexation et le cliquet, ou suit-il l'IPC (**indexé**) ? Quand l’IPC dépasse le drag précédent, les trois lectures coïncident. Le début de l’historique contient toutefois un précédent où l’IPC est inférieur au drag précédent ; la cellule suivante le montre. Ce précédent isolé, hors du régime récent, ne suffit pas à identifier la politique future.
#
# **Pondération : uniforme.** Les données ne permettent pas de départager solidement ces lectures ; nous leur donnons le même poids. Cette grille et sa pondération restent des choix de jugement, même uniformes. La section 11b publie la réponse sous les lectures « règles observées » (la loi, des concessions indexées) pour que l'on voie ce que pèse ce choix. Nous **n'ajoutons pas** de dimension pour les relocations ni pour la part de renouvellements : leur variabilité est déjà mesurée par l'erreur du backtest, qui s'ajoute à la fin (section 11c) ; l'ajouter deux fois élargirait la bande sans raison.

# %%
# CONFIG (mélange)
@dataclass(frozen=True)
class MixtureConfig:
    renewal_readings: tuple = ("pourcentage de base pour tous", "daté par l'avis (loi)", "ancienne méthode pour tous")
    concession_paths: tuple = ("indexées sur l'IPC", "à mi-chemin", "cliquet")
    draws_per_run: int = 400          # tirages Monte-Carlo par combinaison distincte (précision mesurée en 11c)


MIXTURE = MixtureConfig()
LAW_READING, INDEXED_PATH, RATCHET_PATH = MIXTURE.renewal_readings[1], MIXTURE.concession_paths[0], MIXTURE.concession_paths[-1]


def uniform_prior(names):
    """A priori d'entropie maximale : le même poids pour chaque lecture que les données ne départagent pas."""
    return {name: 1 / len(names) for name in names}


def notice_share_before(cohort, reform_date):
    """Part de la fenêtre légale d'avis de chaque unité située avant `reform_date` : avis uniforme dans la fenêtre (retenu), ou au plus tôt / au plus tard (bornes)."""
    long_lease = cohort.term_months.to_numpy() >= CONFIG.months_per_year
    earliest_months = np.where(long_lease, FORECAST.notice_months_long_lease[1], FORECAST.notice_months_short_lease[1])
    latest_months = np.where(long_lease, FORECAST.notice_months_long_lease[0], FORECAST.notice_months_short_lease[0])
    earliest = pd.Series([end - pd.DateOffset(months=int(months)) for end, months in zip(cohort.lease_end, earliest_months)])
    latest = pd.Series([end - pd.DateOffset(months=int(months)) for end, months in zip(cohort.lease_end, latest_months)])
    if FORECAST.notice_timing == "earliest":
        return (earliest < reform_date).to_numpy(dtype=float)
    if FORECAST.notice_timing == "latest":
        return (latest < reform_date).to_numpy(dtype=float)
    return np.clip(((reform_date - earliest).dt.days / (latest - earliest).dt.days).to_numpy(), 0, 1)



def renewal_readings(external_tables, context, cohort, year, cutoff):
    """Ancrage de renouvellement de chaque unité selon trois lectures (en %) ; elles coïncident avant l'année du changement de méthode."""
    window_year = (cohort.next_start.dt.year - starts_in_previous_tal_window(cohort.next_start)).to_numpy()
    fallback = float(context["annual"].renew_qc.dropna().iloc[-1])      # aucun taux connu (début d'historique) : on reconduit les renouvellements
    old_rate = {wy: tal_old_method_rate(external_tables, wy, cutoff) for wy in np.unique(window_year)}
    old_by_unit = np.nan_to_num(np.array([old_rate[wy] for wy in window_year]), nan=fallback)
    if year < FORECAST.tal_new_format_first_year:
        return {name: old_by_unit for name in MIXTURE.renewal_readings}, np.ones(len(cohort))
    new_rate = {wy: tal_base_percentage(external_tables, wy, cutoff) for wy in np.unique(window_year)}
    new_by_unit = np.array([new_rate[wy] if pd.notna(new_rate[wy]) else old_rate[wy] for wy in window_year])
    share_old = notice_share_before(cohort, year_start(FORECAST.tal_new_format_first_year))
    base_for_all = np.full(len(cohort), tal_base_percentage(external_tables, year, cutoff))
    readings = dict(zip(MIXTURE.renewal_readings, (base_for_all, share_old * old_by_unit + (1 - share_old) * new_by_unit, old_by_unit)))
    return readings, share_old


def concession_paths(lagged_cpi, last_drag):
    """Niveau du drag des nouveaux baux selon trois lectures ; identiques quand l'IPC est au-dessus du drag de l'an dernier."""
    ratchet = max(lagged_cpi, last_drag)
    return dict(zip(MIXTURE.concession_paths, (lagged_cpi, (lagged_cpi + ratchet) / 2, ratchet)))


MEDIAN_PERCENTILE = 100 / 2


def weighted_percentile(values, weights, percentile):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if values.ndim != 1 or values.shape != weights.shape or not values.size:
        raise ValueError("Valeurs et poids doivent être des vecteurs non vides de même longueur")
    if not np.isfinite(values).all() or not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Valeurs et poids doivent être finis ; poids positifs ou nuls")
    if not np.isfinite(percentile) or not 0 <= percentile <= 100 or weights.sum() <= 0:
        raise ValueError("Percentile hors intervalle ou masse totale nulle")
    positive = weights > 0
    values, weights = values[positive], weights[positive]
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order])
    position = np.searchsorted(cumulative, percentile / 100 * cumulative[-1], side="left")
    return float(values[order[min(position, len(order) - 1)]])



def forecast_mixture(leases_table, asking_table, external_tables, year, vintage, n_draws=None, seed=CONFIG.random_seed, priors=None, segment_keys=()):
    """Modèle structurel : lectures de renouvellement × trajectoires de concessions (a priori uniforme par défaut) ; une simulation par combinaison distincte.
    Mêmes nombres aléatoires pour toutes les combinaisons : les écarts entre scénarios ne viennent que des entrées."""
    renewal_weights, concession_weights = priors or (uniform_prior(MIXTURE.renewal_readings), uniform_prior(MIXTURE.concession_paths))
    for selected, allowed in ((renewal_weights, MIXTURE.renewal_readings), (concession_weights, MIXTURE.concession_paths)):
        if not selected or set(selected) - set(allowed):
            raise ValueError("La grille contient une lecture inconnue ou aucune lecture")
        probabilities = np.asarray(list(selected.values()), dtype=float)
        if not np.isfinite(probabilities).all() or (probabilities < 0).any() or not np.isclose(probabilities.sum(), 1):
            raise ValueError("Les a priori doivent être finis, non négatifs et sommer à un")
    if n_draws is not None and (not isinstance(n_draws, int) or isinstance(n_draws, bool) or n_draws <= 0):
        raise ValueError("Le nombre de tirages doit être un entier strictement positif")
    context = prepare_context(*known_at(leases_table, asking_table, year))
    cohort = build_cohort(context, year)
    cutoff = vintage_cutoff(year, vintage)
    readings, share_old = renewal_readings(external_tables, context, cohort, year, cutoff)
    lagged_cpi = cpi_driver(external_tables, year - 1, cutoff)
    last_drag = float(context["annual"].drag.loc[year - 1])
    paths = concession_paths(lagged_cpi, last_drag)
    runs, rows = {}, []
    for (renewal_name, renewal_weight), (concession_name, concession_weight) in product(renewal_weights.items(), concession_weights.items()):
        key = (tuple(np.round(readings[renewal_name], CONFIG.fine_digits)), round(paths[concession_name], CONFIG.fine_digits))
        if key not in runs:
            simulation = simulate_cohort(context, cohort, year, readings[renewal_name], lagged_cpi, paths[concession_name], n_draws or MIXTURE.draws_per_run, seed)
            runs[key] = summarise_simulation(simulation, segment_keys)
        rows.append({"renouvellements": renewal_name, "concessions": concession_name, "poids": renewal_weight * concession_weight, "run": key})
    frame = pd.DataFrame(rows)
    pooled = {statistic: np.concatenate([runs[row.run][statistic] for row in frame.itertuples()])
              for statistic in ("contract", "effective", "renewal_contract", "relocation_contract", "contract_mean", "effective_mean", "concession_effect")}
    weights = np.concatenate([np.full(len(runs[row.run]["contract"]), row.poids / len(runs[row.run]["contract"])) for row in frame.itertuples()])
    quebec = (cohort.province == "Quebec").to_numpy()
    centers = {"renew_qc": float(sum(weight * readings[name][quebec].mean() for name, weight in renewal_weights.items()) / sum(renewal_weights.values())),
               "reloc_qc": lagged_cpi, "drag": float(sum(weight * paths[name] for name, weight in concession_weights.items()) / sum(concession_weights.values()))}
    return {"frame": frame, "runs": runs, "pooled": pooled, "weights": weights, "readings": readings, "share_old": share_old, "paths": paths,
            "lagged_cpi": lagged_cpi, "last_drag": last_drag, "context": context, "cohort": cohort, "n_units": len(cohort), "centers": centers,
            "effective_p50": weighted_percentile(pooled["effective"], weights, MEDIAN_PERCENTILE),
            "contract_p50": weighted_percentile(pooled["contract"], weights, MEDIAN_PERCENTILE)}

# %% [markdown]
# ### 10e. `estimate_2026()` et `backtest()`
# `estimate_2026` retourne la hausse 2026 de la **définition principale** (loyer effectif, médiane des paires) en pourcentage, comme le gabarit le demande : la médiane du modèle structurel. `backtest` rejoue **exactement la même fonction** (`forecast_mixture`) comme à la fin de `target_year − 1` : les baux et les loyers affichés sont tronqués à cette date *avant* tout calcul, l'information publique est filtrée par sa date de publication, puis le résultat est comparé à ce qui s'est réellement passé. `forecast_detail` produit le comparateur générique (combinaison de règles).

# %%
def estimate_2026(leases, asking, external=None):
    """Hausse 2026 (en %) : croissance à unité constante du loyer EFFECTIF, médiane des paires, renouvellements + relocations (modèle structurel)."""
    tables = external if external is not None else EXTERNAL_TABLES
    return forecast_mixture(leases, asking, tables, CONFIG.forecast_year, FORECAST.live_vintage)["effective_p50"]


def forecast_detail(leases_table, asking_table, external_tables, year, vintage, n_draws, seed=CONFIG.random_seed):
    """Comparateur générique : centres = combinaison des règles pondérée par l'erreur, puis même simulation de cohorte."""
    context = prepare_context(*known_at(leases_table, asking_table, year))
    components = {}
    for component in ("renew_qc", "reloc_qc", "drag"):
        center, weights, rules = ensemble_forecast(context, external_tables, component, year, vintage)
        components[component] = {"center": center, "weights": weights, "rules": rules}
    centers = {name: values["center"] for name, values in components.items()}
    cohort = build_cohort(context, year)
    summary = summarise_simulation(simulate_cohort(context, cohort, year, centers["renew_qc"], centers["reloc_qc"], centers["drag"], n_draws, seed))
    return {"components": components, "centers": centers, "contract_median": float(np.median(summary["contract"])),
            "effective_median": float(np.median(summary["effective"])), "n_units": len(cohort)}


def realised_growth(leases_table, year):
    """Valeur réalisée de la cible pour `year`, par exactement les mêmes définitions que la prévision."""
    valid = build_pairs(leases_table.sort_values(UNIT_KEY + ["lease_start"]).reset_index(drop=True))
    valid = valid[valid.is_valid & (valid.lease_year == year)]
    quebec = valid[valid.province == "Quebec"]
    return {"contract": valid.growth_contract_pct.median(), "effective": valid.growth_effective_pct.median(),
            "renew_qc": quebec[quebec.is_renewal == 1].growth_contract_pct.median(), "reloc_qc": quebec[quebec.is_renewal == 0].growth_contract_pct.median(),
            "drag": leases_table[leases_table.lease_year == year].drag.mean() * 100, "pairs": len(valid)}


def backtest(leases, target_year, asking_table=None, external=None, vintage="january", model="structural"):
    """Exécute la méthode comme à la fin de target_year − 1 et compare avec ce qui s'est passé en target_year."""
    if model not in ("structural", "ensemble"):
        raise ValueError("Le modèle doit être structural ou ensemble")
    tables = external if external is not None else EXTERNAL_TABLES
    asking_used = asking_table if asking_table is not None else asking
    if model == "structural":
        result = forecast_mixture(leases, asking_used, tables, target_year, vintage, n_draws=FORECAST.n_simulations_backtest)
        predicted = {"contract": result["contract_p50"], "effective": result["effective_p50"]}
    else:
        result = forecast_detail(leases, asking_used, tables, target_year, vintage, FORECAST.n_simulations_backtest)
        predicted = {"contract": result["contract_median"], "effective": result["effective_median"]}
    actual = realised_growth(leases, target_year)
    return {"year": target_year, "vintage": vintage, "model": model, "predicted_effective": predicted["effective"], "actual_effective": actual["effective"],
            "error_effective": predicted["effective"] - actual["effective"], "predicted_contract": predicted["contract"],
            "actual_contract": actual["contract"], "error_contract": predicted["contract"] - actual["contract"],
            "centers": result["centers"], "actual_components": {k: actual[k] for k in ("renew_qc", "reloc_qc", "drag")}}

# %% [markdown]
# ### 10f. Le backtest : 2023, 2024 et 2025 (et 2021-2022 pour plus de points)
# Pour chaque année cible et chaque millésime : la prévision, la valeur réalisée, l'erreur. **Références toujours affichées** : *reconduire l'an dernier*, *moyenne des dernières années*, *dérive* (dernière valeur + variation moyenne récente) et *réglementaire seulement* (TAL pour les renouvellements, dernière relocation pour les relocations, mélange à la part de renouvellement récente).
#
# **Statut statistique, dit honnêtement.** Les règles du modèle structurel n'ont **aucun paramètre ajusté sur l'année prédite** (la pente du TAL et la sensibilité `k` de la pénétration sont estimées à chaque date de coupure, sur le seul passé), mais elles ont été **choisies** en regardant 2018-2025 : pour le choix de la structure, le backtest est donc **dans l'échantillon**. Le comparateur générique, lui, est réellement hors échantillon.

# %%
MODEL_LABELS = {"structural": "modèle structurel (retenu)", "ensemble": "combinaison de règles"}
backtest_table = pd.DataFrame([backtest(leases, target_year, vintage=vintage, model=model)
                               for model in MODEL_LABELS for vintage in FORECAST.vintage_dates for target_year in CONFIG.extended_backtest_years])


def baseline_forecasts(year):
    """Références simples calculées à partir des seules années antérieures à `year`."""
    known = prepare_context(*known_at(leases, asking, year))
    annual_series = known["annual"]
    out = {}
    for kind, column in (("contract", "all_contract"), ("effective", "all_effective")):
        history = annual_series[column].loc[:year - 1].dropna()
        out[f"reconduire ({kind})"] = history.iloc[-1]
        out[f"moyenne récente ({kind})"] = history.tail(CONFIG.recent_years_for_mean).mean()
        out[f"dérive ({kind})"] = history.iloc[-1] + history.diff().dropna().tail(CONFIG.recent_years_for_mean).mean()
    regulatory = tal_old_method_rate(EXTERNAL_TABLES, year, vintage_cutoff(year, "january"))
    renewal_share_last = known["pairs"][known["pairs"].lease_year == year - 1].is_renewal.mean()
    out["réglementaire seul (contract)"] = renewal_share_last * regulatory + (1 - renewal_share_last) * annual_series.reloc_qc.loc[:year - 1].iloc[-1]
    return out


baseline_rows = []
for target_year in CONFIG.extended_backtest_years:
    truth = realised_growth(leases, target_year)
    for name, value in baseline_forecasts(target_year).items():
        kind = "effective" if "(effective)" in name else "contract"
        baseline_rows.append({"year": target_year, "method": name.split(" (")[0], "kind": kind, "predicted": value, "actual": truth[kind], "error": value - truth[kind]})
baseline_table = pd.DataFrame(baseline_rows)

model_summary = backtest_table.melt(id_vars=["year", "vintage", "model"], value_vars=["error_contract", "error_effective"], var_name="kind", value_name="error")
model_summary["kind"] = model_summary.kind.str.replace("error_", "")
model_summary["method"] = model_summary.model.map(MODEL_LABELS) + " (" + model_summary.vintage + ")"
all_errors = pd.concat([model_summary[["year", "method", "kind", "error"]], baseline_table[["year", "method", "kind", "error"]]])


def error_summary(frame, years):
    selected = frame[frame.year.isin(years)]
    return selected.groupby(["kind", "method"]).error.agg(MAE=lambda e: e.abs().mean(), biais="mean", pire_erreur=lambda e: e.abs().max())


required_summary = error_summary(all_errors, CONFIG.backtest_years)
extended_summary = error_summary(all_errors, CONFIG.extended_backtest_years)
print(f"=== Années exigées par le défi : {CONFIG.backtest_years} ===")
display(required_summary.round(CONFIG.table_digits))
print(f"=== Backtest étendu {CONFIG.extended_backtest_years[0]}-{CONFIG.extended_backtest_years[-1]} ({len(CONFIG.extended_backtest_years)} points) ===")
display(extended_summary.round(CONFIG.table_digits))

# %%
live_rows = backtest_table[(backtest_table.vintage == FORECAST.live_vintage) & (backtest_table.model == "structural")].set_index("year")
detail_columns = ["predicted_contract", "actual_contract", "error_contract", "predicted_effective", "actual_effective", "error_effective"]
display(live_rows[detail_columns].round(CONFIG.table_digits))
component_table = pd.DataFrame({(component, side): {year: (row.centers[key] if side == "prévu" else row.actual_components[key]) for year, row in live_rows.iterrows()}
                                for component, key in (("renouvellements QC", "renew_qc"), ("relocations QC", "reloc_qc"), ("drag (%)", "drag"))
                                for side in ("prévu", "réalisé")})
display(component_table.round(CONFIG.table_digits))

fig, axes = plt.subplots(1, 2, figsize=STYLE.figsize_wide)
ensemble_rows = backtest_table[(backtest_table.vintage == FORECAST.live_vintage) & (backtest_table.model == "ensemble")].set_index("year")
for ax, kind, title in zip(axes, ("contract", "effective"), ("contractuel", "effectif")):
    ax.plot(live_rows.index, live_rows[f"actual_{kind}"], "ko-", label="réalisé")
    ax.plot(live_rows.index, live_rows[f"predicted_{kind}"], "s--", color=PALETTE[kind], label=f"{MODEL_LABELS['structural']} ({FORECAST.live_vintage})")
    ax.plot(ensemble_rows.index, ensemble_rows[f"predicted_{kind}"], "^:", color=PALETTE["alternative"], label=f"{MODEL_LABELS['ensemble']} ({FORECAST.live_vintage})")
    persist = baseline_table[(baseline_table.method == "reconduire") & (baseline_table.kind == kind)].set_index("year").predicted
    ax.plot(persist.index, persist, ":", color=PALETTE["naive"], label="reconduire l'an dernier")
    ax.set(title=f"Backtest, hausse {title} (%)")
    ax.legend(fontsize=STYLE.legend_fontsize)
plt.tight_layout()
plt.show()

# %%
def mae_of(method, kind, years):
    return float(all_errors[(all_errors.method == method) & (all_errors.kind == kind) & all_errors.year.isin(years)].error.abs().mean())


structural_live = f"{MODEL_LABELS['structural']} ({FORECAST.live_vintage})"
structural_december = f"{MODEL_LABELS['structural']} (december)"
ensemble_live_label = f"{MODEL_LABELS['ensemble']} ({FORECAST.live_vintage})"
naive_methods = ["reconduire", "moyenne récente", "dérive"]
naive_mae = {kind: [mae_of(method, kind, CONFIG.backtest_years) for method in naive_methods] for kind in ("contract", "effective")}
early_years = [year for year in CONFIG.extended_backtest_years if year not in CONFIG.backtest_years]
narrate(f"**Lecture du backtest** (chiffres exacts dans les tableaux ci-dessus).\n"
        f"* **Le modèle structurel** se trompe en moyenne de **{fmt(mae_of(structural_live, 'contract', CONFIG.backtest_years), CONFIG.table_digits)} point (contractuel) et "
        f"{fmt(mae_of(structural_live, 'effective', CONFIG.backtest_years), CONFIG.table_digits)} point (effectif)** sur {CONFIG.backtest_years[0]}-{CONFIG.backtest_years[-1]}, "
        f"contre {fmt(min(naive_mae['effective']), CONFIG.table_digits)} à {fmt(max(naive_mae['effective']), CONFIG.table_digits)} point pour les références naïves (effectif) et "
        f"{fmt(mae_of(ensemble_live_label, 'effective', CONFIG.backtest_years), CONFIG.table_digits)} pour la combinaison de règles.\n"
        f"* **Hors du régime récent** ({', '.join(map(str, early_years))}), son erreur monte à {fmt(mae_of(structural_live, 'contract', early_years), CONFIG.table_digits)} point (contractuel) : "
        "les renouvellements dépassaient alors le TAL (section 6b) et le régime d'indexation des concessions n'existait pas encore. Le modèle est **bon dans le régime actuel, pas universel** : "
        "la section 11c chiffre ce risque de régime à part.\n"
        f"* **Le millésime décembre** prévoit le TAL de l'année cible **avant sa publication** par sa formule (section 10c) ; son erreur effective "
        f"({fmt(mae_of(structural_december, 'effective', CONFIG.backtest_years), CONFIG.table_digits)}) se compare à celle du millésime en direct.\n"
        "* **Prudence :** la structure a été choisie en regardant ces années (backtest dans l'échantillon pour le choix du modèle). Ce qui la rend crédible : aucun paramètre "
        "ajusté pour le niveau du drag et les relocations, deux paramètres seulement estimés à chaque date de coupure sur le passé (la pente du TAL avant sa publication, la "
        "sensibilité `k` de la pénétration), un mécanisme plausible (budget de concessions réinitialisé chaque janvier sur l'inflation des loyers) et, pour le TAL, une formule "
        "publique vérifiable. La combinaison de règles, réellement hors échantillon, montre ce qu'on obtient sans cette structure.")

# %%
backtest_headline = pd.DataFrame({
    f"MAE {CONFIG.backtest_years[0]}-{CONFIG.backtest_years[-1]}": required_summary.MAE,
    f"MAE {CONFIG.extended_backtest_years[0]}-{CONFIG.extended_backtest_years[-1]}": extended_summary.MAE}).unstack("kind")
print("Synthèse du backtest (erreur absolue moyenne, en points) :")
backtest_headline.round(CONFIG.table_digits)

# %% [markdown]
# ### 10f bis. Robustesse : chaque réglage déclaré, testé
# Chaque réglage qui n'est pas estimé (fenêtre de `k`, fenêtre de la part de renouvellement, distribution des concessions, date de l'avis dans la fenêtre légale) est remplacé par ses alternatives raisonnables (`ForecastConfig.robustness_variants`). Pour chacune : la réponse 2026 et l'erreur du backtest 2023-2025 (millésime en direct). **Ce tableau ne sert pas à choisir le meilleur réglage** (ce serait régler le modèle sur les années testées) : il montre que la réponse n'en dépend pas, ou dit de combien elle en dépend.

# %%
def robustness_row(changes):
    """Réponse 2026 et erreur du backtest 2023-2025 sous des réglages modifiés (rétablis ensuite)."""
    with forecast_settings(**changes):
        forecast = forecast_mixture(leases, asking, EXTERNAL_TABLES, CONFIG.forecast_year, FORECAST.live_vintage)
        replays = [backtest(leases, year, vintage=FORECAST.live_vintage) for year in CONFIG.backtest_years]
    return {f"effectif {CONFIG.forecast_year} (médiane)": forecast["effective_p50"], f"contractuel {CONFIG.forecast_year} (médiane)": forecast["contract_p50"],
            "MAE effectif 2023-2025": np.mean([abs(replay["error_effective"]) for replay in replays]),
            "MAE contractuel 2023-2025": np.mean([abs(replay["error_contract"]) for replay in replays])}


robustness = pd.DataFrame({"réglages retenus": robustness_row({}),
                           **{label: robustness_row(changes) for label, changes in FORECAST.robustness_variants.items()}}).T
robustness["écart effectif (points)"] = robustness.iloc[:, 0] - robustness.iloc[0, 0]
robustness["écart contractuel (points)"] = robustness.iloc[:, 1] - robustness.iloc[0, 1]
robustness.round(CONFIG.table_digits)

# %%
def material_settings(shift_column, error_column):
    """Réglages qui déplacent la réponse de plus que l'erreur du backtest du modèle retenu (le seuil vient du modèle lui-même)."""
    threshold = robustness.loc["réglages retenus", error_column]
    return robustness[robustness[shift_column].abs() > threshold], threshold


def describe_material(rows, shift_column, error_column, threshold, label):
    if rows.empty:
        return f"* **{label} :** aucun réglage ne déplace la réponse de plus que l'erreur du backtest ({fmt(threshold, CONFIG.table_digits)} point)."
    details = " ; ".join(f"« {name} » {fmt(row[shift_column], CONFIG.table_digits, signed=True)} point (erreur du backtest {fmt(row[error_column], CONFIG.table_digits)})"
                         for name, row in rows.iterrows())
    return f"* **{label} :** au-delà de l'erreur du backtest ({fmt(threshold, CONFIG.table_digits)} point) : {details}."


material_effective, threshold_effective = material_settings("écart effectif (points)", "MAE effectif 2023-2025")
material_contract, threshold_contract = material_settings("écart contractuel (points)", "MAE contractuel 2023-2025")
narrate(f"**Lecture.** Un réglage compte s'il déplace la réponse de plus que l'erreur du backtest du modèle retenu. Sur {len(robustness) - 1} réglages alternatifs, "
        f"la réponse effective {CONFIG.forecast_year} reste entre {fmt(robustness.iloc[:, 0].min(), CONFIG.table_digits)} et {fmt(robustness.iloc[:, 0].max(), CONFIG.table_digits)} % "
        f"(retenu : {fmt(robustness.iloc[0, 0], CONFIG.table_digits)} %), la contractuelle entre {fmt(robustness.iloc[:, 1].min(), CONFIG.table_digits)} et "
        f"{fmt(robustness.iloc[:, 1].max(), CONFIG.table_digits)} %.\n"
        + describe_material(material_effective, "écart effectif (points)", "MAE effectif 2023-2025", threshold_effective, "Effectif") + "\n"
        + describe_material(material_contract, "écart contractuel (points)", "MAE contractuel 2023-2025", threshold_contract, "Contractuel") + "\n\n"
        "Pourquoi garder les réglages retenus : `k` est estimé sur les années récentes parce qu'il dérive dans le temps (section 10c) ; l'estimer sur tout l'historique "
        "mélange l'ancien régime et dégrade le backtest. L'avis « uniforme » est le milieu de la fenêtre légale ; les deux bornes (au plus tôt, au plus tard) encadrent "
        "la réponse, et la date de l'avis ne change que 2026 (aucun changement de méthode avant), d'où un backtest identique.")

# %%
# Contrôles de validité : pas de fuite d'information -----------------------------------------------------------
first_backtest_year = CONFIG.backtest_years[0]
context_first = prepare_context(*known_at(leases, asking, first_backtest_year))
assert context_first["leases"].lease_start.max() <= year_end(first_backtest_year - 1), "aucun bail postérieur à la coupure"
assert context_first["annual"].index.max() <= first_backtest_year - 1, "aucune année cible dans l'historique"
assert vintage_cutoff(first_backtest_year, "december") < vintage_cutoff(first_backtest_year, "january") < vintage_cutoff(first_backtest_year, "october")
assert first_backtest_year not in published_series(EXTERNAL_TABLES, "tal_recommended_unheated_old_method", vintage_cutoff(first_backtest_year, "december")).index, "TAL inconnu en décembre"
assert first_backtest_year in published_series(EXTERNAL_TABLES, "tal_recommended_unheated_old_method", vintage_cutoff(first_backtest_year, "january")).index, "TAL connu en janvier"
assert set(backtest_table.year) == set(CONFIG.extended_backtest_years)
print("Contrôles de la section 10 : OK")
