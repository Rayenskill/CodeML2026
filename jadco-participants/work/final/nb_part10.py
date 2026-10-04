# %% [markdown]
# ### 10g. Les loyers affichés disent-ils quelque chose sur 2026 ? (et pourquoi les `listings` n'aident pas)
# Intuition séduisante : « nous affichons, nous signons un peu sous l'affichage, donc les relocations de 2026 sont déjà visibles dans les affichages actuels ». Nous la testons au lieu de la croire. **Étape qui crée deux tables** (`signed`, `last_lease_ask`) : chaque relocation, puis chaque dernier bail, associé au dernier affichage de son unité au plus `ask_match_tolerance_days` jours avant la signature.

# %%
ask_tolerance = pd.Timedelta(days=CONFIG.ask_match_tolerance_days)
ask_columns = asking[UNIT_KEY + ["ask_date", "rent_ask"]].sort_values("ask_date")
signed = pd.merge_asof(
    leases[leases.is_renewal == 0].assign(sign_month=lambda d: d.sign_date.dt.to_period("M").dt.to_timestamp()).sort_values("sign_month"),
    ask_columns, left_on="sign_month", right_on="ask_date", by=UNIT_KEY, direction="backward", tolerance=ask_tolerance)
signed["signed_vs_ask_pct"] = (signed.rent_contract / signed.rent_ask - 1) * 100
discount_by_year = signed.groupby("lease_year").signed_vs_ask_pct.agg(relocations_matched="count", median_signed_vs_ask="median").loc[CONFIG.first_year:]
display(discount_by_year.round(CONFIG.table_digits))
last_lease_ask = (leases.sort_values("lease_start").groupby(UNIT_KEY).tail(1)
                  .assign(sign_month=lambda d: d.sign_date.dt.to_period("M").dt.to_timestamp()).sort_values("sign_month"))
last_lease_ask = pd.merge_asof(last_lease_ask, ask_columns, left_on="sign_month", right_on="ask_date", by=UNIT_KEY, direction="backward", tolerance=ask_tolerance)
listing_check = units.merge(last_lease_ask[UNIT_KEY + ["rent_contract", "rent_ask"]], on=UNIT_KEY).dropna(subset=["rent_ask"])
listing_vs_current_ask = (listing_check.rent_listed_2025 / listing_check.rent_ask - 1) * 100
listing_vs_current_lease = (listing_check.rent_listed_2025 / listing_check.rent_contract - 1) * 100
listing_level_correlation = np.corrcoef(listing_check.rent_listed_2025, listing_check.rent_ask)[0, 1]
print(f"Loyer du fichier `listings` contre l'affichage ayant produit le bail ACTUEL : médiane {listing_vs_current_ask.median():+.2f} %, corrélation des niveaux {listing_level_correlation:.3f}")
print(f"Loyer du fichier `listings` contre le loyer du bail actuel : médiane {listing_vs_current_lease.median():+.2f} %")

# %%
usual_discount = -discount_by_year.median_signed_vs_ask.median()
narrate(f"**Lecture.**\n"
        f"* **L'escompte de négociation est stable** : les relocations se signent en médiane {fmt(usual_discount)} % sous l'affichage, chaque année entre "
        f"{fmt(discount_by_year.median_signed_vs_ask.min(), signed=True)} et {fmt(discount_by_year.median_signed_vs_ask.max(), signed=True)} %. La règle « signé ≈ affiché moins un petit escompte » est solide.\n"
        f"* **Mais le fichier `listings` n'apporte aucun signal de {CONFIG.forecast_year}.** Son `rent_listed_2025` coïncide avec l'affichage qui a produit le **bail actuel** "
        f"(écart médian {fmt(listing_vs_current_ask.median(), CONFIG.table_digits, signed=True)} %, corrélation des niveaux {fmt(listing_level_correlation, CONFIG.fine_digits)}) "
        f"et dépasse le loyer du bail de {fmt(listing_vs_current_lease.median(), signed=True)} % : c'est exactement l'escompte habituel, pas une hausse annoncée. "
        "Lire cet écart comme une hausse de 2026 serait **une erreur de lecture** : les dates de disponibilité (`available_date`) ne sont pas liées aux fins de bail, "
        "et ce fichier décrit le niveau de 2025, pas un prix 2026.\n"
        "* **Conséquence :** nous n'utilisons pas `listings` comme variable prédictive ; la **variation** des loyers affichés d'Équinoxe (série `ask`) reste une règle candidate "
        "du comparateur (section 10a) et l'escompte sert de preuve de cohérence.")

# %% [markdown]
# ### 10h. Défi de l'apprentissage automatique : XGBoost, gardé seulement s'il bat la méthode structurelle
# L'énoncé permet un modèle de régression « s'il est justifié ». Le risque est connu : quelques milliers de paires, mais seulement **quelques années distinctes** de conditions de marché. Un modèle flexible peut apprendre le bruit propre à chaque année. Nous lui donnons donc toutes ses chances, avec des garde-fous :
# * **variables connues au moment de fixer le loyer** : propriété, chambres, superficie, étage, renouvellement ou non, durée et concession du bail précédent, écart de loyer du bail précédent au niveau de son segment (*loss-to-lease*), et variables de marché (taux TAL connu, IPC décalé, affichages décalés, inoccupation décalée) ;
# * **ni année ni date** (elles ne s'extrapolent pas à 2026) ;
# * **contraintes de monotonie** : la hausse ne diminue pas quand le TAL, l'IPC décalé ou les affichages décalés augmentent ;
# * **validation à origine glissante** : on entraîne sur les années `≤ t−1` et on prédit `t` ; hyperparamètres petits et **fixés avant** de regarder les résultats ;
# * **gardé seulement s'il bat la méthode structurelle** sur le même backtest.

# %%
# CONFIG (apprentissage automatique) : hyperparamètres petits, fixés a priori
XGB_PARAMS = dict(max_depth=3, n_estimators=200, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, min_child_weight=20, reg_lambda=5.0,
                  random_state=CONFIG.random_seed, n_jobs=1, tree_method="hist")
ML_MIN_TRAINING_YEARS = 3          # au moins trois années d'entraînement avant la première année prédite
SHAP_SAMPLE_SIZE = 5000            # paires expliquées par SHAP (au plus)
SHAP_TOP_FEATURES = 3              # variables citées dans la lecture
FEATURE_COLUMNS = ["building_code", "beds", "sqft", "floor", "is_renewal", "prev_term_months", "prev_drag", "loss_to_lease", "tal_rate", "cpi_lag", "ask_lag", "vacancy_lag"]
MONOTONE = {"tal_rate": 1, "cpi_lag": 1, "ask_lag": 1}

# %%
ml_backtest_years = [year for year in CONFIG.extended_backtest_years if year - CONFIG.first_year >= ML_MIN_TRAINING_YEARS]
building_codes = {code: index for index, code in enumerate(sorted(leases.prop_code.unique()))}


def market_features(year, vintage=FORECAST.live_vintage):
    """Variables de marché de l'année `year`, telles que connues à la date de coupure du millésime (TAL sur l'échelle de l'ancienne méthode)."""
    cutoff = vintage_cutoff(year, vintage)
    asking_previous = same_unit_asking_growth(asking[asking.ask_date <= year_end(year - 1)]).groupby("ask_year").growth_pct.median()
    vacancy_known = external[(external.series == "cmhc_vacancy_rate") & (external.geo == "Montreal CMA") & (external.published_on <= cutoff)].set_index("year").value
    return {"tal_rate": tal_old_method_rate(EXTERNAL_TABLES, year, cutoff), "cpi_lag": cpi_driver(EXTERNAL_TABLES, year - 1, cutoff),
            "ask_lag": float(asking_previous.get(year - 1, np.nan)), "vacancy_lag": float(vacancy_known.get(year - 1, np.nan))}


def segment_median_rent(year):
    """Loyer contractuel médian des baux ayant débuté en `year`, par propriété × chambres."""
    return leases[leases.lease_year == year].groupby(["prop_code", "beds"]).rent_contract.median()


def feature_table(rows, year, previous_rent, previous_drag, previous_term, renewal_flag):
    """Variables d'apprentissage pour des baux de l'année `year` (aucune variable ex post)."""
    medians = segment_median_rent(year - 1)
    segment_level = np.array([medians.get(key, np.nan) for key in zip(rows.prop_code, rows.beds)])
    table = pd.DataFrame({
        "building_code": rows.prop_code.map(building_codes).to_numpy(), "beds": rows.beds.to_numpy(), "sqft": rows.sqft.to_numpy(),
        "floor": rows.floor.to_numpy(), "is_renewal": np.asarray(renewal_flag, dtype=float), "prev_term_months": np.asarray(previous_term, dtype=float),
        "prev_drag": np.asarray(previous_drag, dtype=float), "loss_to_lease": np.log(segment_level) - np.log(np.asarray(previous_rent, dtype=float))})
    for name, value in market_features(year).items():
        table[name] = value
    return table[FEATURE_COLUMNS]


def training_set(last_training_year):
    """Paires des années first_year..last_training_year avec leurs variables ex ante et la cible contractuelle."""
    frames, targets = [], []
    for year in range(CONFIG.first_year, last_training_year + 1):
        subset = pairs[pairs.lease_year == year]
        frames.append(feature_table(subset, year, subset.prev_rent_contract, subset.prev_drag, subset.prev_term_months, subset.is_renewal))
        targets.append(subset.growth_contract_pct.to_numpy())
    features = pd.concat(frames, ignore_index=True)
    complete = features.notna().all(axis=1).to_numpy()
    return features[complete], np.concatenate(targets)[complete]


def fit_model(last_training_year):
    features, target = training_set(last_training_year)
    model = xgboost_library.XGBRegressor(**XGB_PARAMS, monotone_constraints=tuple(MONOTONE.get(name, 0) for name in FEATURE_COLUMNS))
    model.fit(features, target)
    return model


def predict_median_for_cohort(model, context, year):
    """Prévision de la médiane des paires de la cohorte : chaque unité reçoit p × (prévision si renouvellement) + (1 − p) × (prévision si relocation)."""
    cohort = build_cohort(context, year)
    probability = renewal_probability(context, year, cohort)
    common = dict(rows=cohort, year=year, previous_rent=cohort.rent_contract, previous_drag=cohort.drag, previous_term=cohort.term_months)
    as_renewal = model.predict(feature_table(renewal_flag=np.ones(len(cohort)), **common))
    as_relocation = model.predict(feature_table(renewal_flag=np.zeros(len(cohort)), **common))
    return float(np.median(probability * as_renewal + (1 - probability) * as_relocation))


ml_rows = []
for target_year in ml_backtest_years:
    context_ml = prepare_context(*known_at(leases, asking, target_year))
    ml_rows.append({"year": target_year, "XGBoost": predict_median_for_cohort(fit_model(target_year - 1), context_ml, target_year),
                    "actual_contract": realised_growth(leases, target_year)["contract"]})
ml_backtest = pd.DataFrame(ml_rows).set_index("year")
for model_name, label in MODEL_LABELS.items():
    ml_backtest[label] = backtest_table[(backtest_table.vintage == FORECAST.live_vintage) & (backtest_table.model == model_name)].set_index("year").predicted_contract.reindex(ml_backtest.index)
ml_mae = {name: (ml_backtest[name] - ml_backtest.actual_contract).abs().mean() for name in ["XGBoost", *MODEL_LABELS.values()]}
display(ml_backtest.round(CONFIG.table_digits))
print(f"Erreur absolue moyenne (contractuel), {ml_backtest_years[0]}-{ml_backtest_years[-1]} :", {name: round(value, CONFIG.table_digits) for name, value in ml_mae.items()})

# %%
narrate(f"**Verdict.** Hors période, XGBoost se trompe en moyenne de {fmt(ml_mae['XGBoost'], CONFIG.table_digits)} point sur la hausse contractuelle, contre "
        f"{fmt(ml_mae[MODEL_LABELS['structural']], CONFIG.table_digits)} pour le modèle structurel et {fmt(ml_mae[MODEL_LABELS['ensemble']], CONFIG.table_digits)} pour la combinaison de règles "
        f"({ml_backtest_years[0]}-{ml_backtest_years[-1]}). Le modèle à gradient n'a pas assez d'années distinctes de marché pour apprendre les effets d'année : "
        "**nous ne le retenons pas comme prévision**, mais nous le conservons comme **défi indépendant** et nous livrons le modèle entraîné jusqu'à la dernière année observée "
        "(`outputs/model_2026.json`). L'explication SHAP ci-dessous dit ce qu'il a appris.")

# %% [markdown]
# **Le modèle final et ce qu'il a appris.** On entraîne XGBoost sur toutes les années observées, on l'applique à la cohorte 2026 (pour mémoire) et on mesure l'importance de chaque variable (SHAP si la librairie est installée, sinon le gain de XGBoost).

# %%
final_model = fit_model(CONFIG.last_observed_year)
final_model.save_model(CONFIG.output_dir / "model_2026.json")
context_live = prepare_context(leases, asking)
market_live = market_features(CONFIG.forecast_year)
xgboost_2026 = predict_median_for_cohort(final_model, context_live, CONFIG.forecast_year)
print(f"XGBoost, hausse contractuelle {CONFIG.forecast_year} (médiane de la cohorte) : {xgboost_2026:.2f} %   | variables de marché : { {k: round(v, CONFIG.table_digits) for k, v in market_live.items()} }")
print(f"(Le modèle a été entraîné sur l'échelle de l'ancienne méthode du TAL : il reçoit l'équivalent ancienne méthode ({market_live['tal_rate']:.2f} %), "
      f"pas le pourcentage de base ({TAL_NEW_FORMAT[CONFIG.forecast_year]:.1f} %).)")
if shap is not None:
    training_features, _ = training_set(CONFIG.last_observed_year)
    shap_values = shap.TreeExplainer(final_model).shap_values(training_features.sample(min(len(training_features), SHAP_SAMPLE_SIZE), random_state=CONFIG.random_seed))
    shap_importance = pd.Series(np.abs(shap_values).mean(axis=0), index=FEATURE_COLUMNS).sort_values(ascending=False)
    display(shap_importance.round(CONFIG.fine_digits).rename("importance SHAP moyenne (points de %)").to_frame().T)
else:
    shap_importance = pd.Series(final_model.feature_importances_, index=FEATURE_COLUMNS).sort_values(ascending=False)
    print("SHAP indisponible : importance par gain de XGBoost à la place")

# %%
market_names = {"tal_rate", "cpi_lag", "ask_lag", "vacancy_lag"}
market_share = shap_importance[shap_importance.index.isin(market_names)].sum() / shap_importance.sum()
narrate(f"**Lecture.** Les variables les plus importantes sont {', '.join(f'`{name}`' for name in shap_importance.index[:SHAP_TOP_FEATURES])} ; les variables **de marché** "
        f"({', '.join(f'`{name}`' for name in sorted(market_names))}) portent {fmt(market_share * 100, 0)} % de l'importance totale. Cela confirme la structure par composantes, "
        "mais c'est aussi un **avertissement** : avec peu d'années de marché, des variables « de marché » prennent la place d'un effet d'année, et c'est précisément ce qui nuit "
        "à sa précision hors période. Le modèle ne trouve pas de signal caché au niveau de l'unité.")

# %% [markdown]
# ### 10i. 2026 : le modèle structurel en direct, et ce que pèsent les concessions
# Le modèle structurel est calculé **une fois** pour 2026 (millésime en direct), avec les médianes par propriété et par propriété × chambres ; la section 11 réutilise ce calcul. Le tableau suivant isole l'effet des concessions : même ancrage de renouvellement (daté par l'avis), trois trajectoires du drag (section 10d) et, pour mémoire, la **tendance extrapolée** de la combinaison de règles (rejetée : elle n'a pas de mécanisme).

# %%
SEGMENT_KEYS = (("prop_code",), ("prop_code", "beds"))
mixture = forecast_mixture(leases, asking, EXTERNAL_TABLES, CONFIG.forecast_year, FORECAST.live_vintage, segment_keys=SEGMENT_KEYS)
ensemble_live = forecast_detail(leases, asking, EXTERNAL_TABLES, CONFIG.forecast_year, FORECAST.live_vintage, CONFIG.n_simulations)


def run_scenario(renewal_reading, drag_level, ontario_regime="exempt", ontario_guideline=None, n_draws=CONFIG.n_simulations):
    """Une simulation de la cohorte 2026 avec une lecture de renouvellement et un niveau de drag donnés (mêmes nombres aléatoires partout)."""
    return simulate_cohort(mixture["context"], mixture["cohort"], CONFIG.forecast_year, mixture["readings"][renewal_reading], mixture["lagged_cpi"], drag_level,
                           n_draws, CONFIG.random_seed, ontario_regime=ontario_regime, ontario_guideline=ontario_guideline)


drag_levels = {**mixture["paths"], "tendance extrapolée (rejetée)": ensemble_live["components"]["drag"]["rules"]["tendance"]}
concession_rows = []
for name, level in drag_levels.items():
    summary = summarise_simulation(run_scenario(LAW_READING, level))
    concession_rows.append({"trajectoire": name, "drag 2026 (%)": level, "contractuel (médiane)": np.median(summary["contract"]),
                            "effectif (médiane)": np.median(summary["effective"]), "effectif (moyenne pondérée)": np.median(summary["effective_mean"])})
concession_table = pd.DataFrame(concession_rows).set_index("trajectoire")
concession_table.round(CONFIG.table_digits)

# %%
drag_sensitivity = ((concession_table.loc[INDEXED_PATH, "effectif (médiane)"] - concession_table.loc[RATCHET_PATH, "effectif (médiane)"])
                    / (concession_table.loc[RATCHET_PATH, "drag 2026 (%)"] - concession_table.loc[INDEXED_PATH, "drag 2026 (%)"]))
narrate(f"**Lecture.** À hausse contractuelle donnée ({fmt(concession_table['contractuel (médiane)'].iloc[0])} %), l'effectif {CONFIG.forecast_year} dépend presque uniquement "
        f"de la trajectoire des concessions : chaque point de drag supplémentaire sur les nouveaux baux retire **{fmt(drag_sensitivity, CONFIG.table_digits)} point** à la hausse effective. "
        f"Indexées sur l'IPC ({fmt(concession_table.loc[INDEXED_PATH, 'drag 2026 (%)'])} %), les concessions donnent {fmt(concession_table.loc[INDEXED_PATH, 'effectif (médiane)'])} % ; "
        f"en cliquet ({fmt(concession_table.loc[RATCHET_PATH, 'drag 2026 (%)'])} %), {fmt(concession_table.loc[RATCHET_PATH, 'effectif (médiane)'])} %. "
        f"La tendance extrapolée ({fmt(drag_levels['tendance extrapolée (rejetée)'])} %) donnerait {fmt(concession_table.loc['tendance extrapolée (rejetée)', 'effectif (médiane)'])} % : "
        "rejetée, parce qu'elle prolonge une hausse sans mécanisme.")

# %%
# Contrôles de validité ----------------------------------------------------------------------------------------
assert concession_table.loc[INDEXED_PATH, "effectif (médiane)"] > concession_table.loc[RATCHET_PATH, "effectif (médiane)"], "moins de drag (indexé) = plus d'effectif que le cliquet"
assert np.isfinite(xgboost_2026), "prévision XGBoost finie"
assert (CONFIG.output_dir / "model_2026.json").exists()
print("Contrôles de la section 10 (suite) : OK")
