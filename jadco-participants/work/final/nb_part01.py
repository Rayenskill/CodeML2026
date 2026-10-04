# %% [markdown]
# # Collection Équinoxe : estimer la hausse de loyer 2026
#
# **Équipe CodeML 2026 · défi JADCO « Clés en main »** · notebook principal (livrable évalué)
#
# ## Réponse en une phrase
# *(La valeur est calculée en section 11 ; aucun chiffre de ce notebook n'est tapé à la main dans le texte.)*
#
# > **Hausse 2026 = croissance annualisée du loyer effectif (net des concessions) à unité constante, mesurée sur les baux qui débutent en 2026 contre le bail précédent de la même unité (`prop_code` + `unit_code`), médiane des paires, renouvellements et relocations confondus.**
# > Nous publions toujours à côté la **hausse contractuelle** (loyer inscrit au bail), seule comparable aux règles du TAL et de l'Ontario.
#
# ## Plan (même ordre que la grille d'évaluation)
# 1. Définir la question et la nommer · 2. Lire les données et renommer les colonnes · 3. Effet de composition (mix) ·
# 4. Appariement à unité constante · 5. Concessions · 6. Renouvellements vs relocations, Québec vs Ontario ·
# 7. Choix de la définition (tableau comparatif) · 8. Données publiques et réconciliation · 9. Estimateurs alternatifs (contre-vérification) ·
# 10. Prévision `estimate_2026()` et `backtest()` · 11. Scénarios, intervalles, immeuble × chambres · 12. Limites et hypothèses · 13. Références (sources et outils d'IA)
#
# ### Conventions de ce notebook (discipline d'un desk quantitatif)
# * **Français** pour les explications, **anglais `snake_case`** pour les identifiants (une seule langue de nommage, tableau de correspondance en section 2).
# * **Aucun nombre magique.** Chaque paramètre est soit **estimé sur les données** (la cellule qui l'estime est montrée), soit **tiré d'une source publique** (tableau `external/`, avec URL et date de publication), soit **déclaré comme a priori** dans une cellule `# CONFIG`, avec sa justification et un test de sensibilité. Hors des cellules `# CONFIG`, les seuls littéraux tolérés sont 0, 1, 2 et 100.
# * **Aucun chiffre tapé dans le texte.** Les paragraphes « Lecture » qui citent des valeurs sont **rédigés par le code** à partir des variables calculées : le texte ne peut pas diverger des données.
# * **Information datée.** Toute donnée publique porte sa date de publication (`published_on`) ; une prévision n'utilise que ce qui était public à sa date de coupure.
# * **Les fichiers sources ne sont jamais modifiés** : tout renommage et toute transformation se font ici, et **aucune donnée CRM n'est exportée** (seuls des agrégats sont écrits sur disque).
# * Une cellule Markdown précède chaque étape qui modifie les données ; un contrôle automatique (`lint_notebook.py`) vérifie toutes ces règles.

# %% [markdown]
# ## 0. Configuration et importations
# Toutes les importations et toutes les constantes sont ici. Chaque constante dit *pourquoi* elle a cette valeur ; celles qui peuvent être estimées le sont plus bas, dans la section qui les utilise.

# %%
# CONFIG
import json
import os
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from itertools import product
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import xgboost as xgboost_library
from IPython.display import Markdown, display
from scipy.stats import norm

try:                       # SHAP et plotly sont des plus : leur absence ne bloque pas le notebook
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="IProgress not found.*")
        import shap
except ImportError:
    shap = None
try:
    import plotly.graph_objects as plotly_go
    from plotly.subplots import make_subplots
except ImportError:
    plotly_go = None

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.width", 170)
pd.set_option("display.max_columns", 40)
pd.set_option("display.float_format", lambda value: f"{value:,.2f}")


@dataclass(frozen=True)
class Config:
    # --- emplacements (le CRM n'est pas livré : l'utilisateur pointe vers ses 4 CSV) ---
    data_dir: Path = Path(os.environ.get("JADCO_DATA_DIR", "../.."))
    external_dir: Path = Path(os.environ.get("JADCO_EXTERNAL_DIR", "external/tidy"))
    output_dir: Path = Path(os.environ.get("JADCO_OUTPUT_DIR", "outputs"))
    data_as_of: str = "2026-10-04"    # date de retrait des données publiques (external/SOURCES.md)
    # --- calendrier ---
    first_year: int = 2019            # première année avec assez de paires pour une médiane stable (comptes en section 4)
    last_observed_year: int = 2025    # fin de l'historique fourni (31 décembre 2025)
    forecast_year: int = 2026         # année à prévoir
    backtest_years: tuple = (2023, 2024, 2025)   # imposé par le défi
    extended_backtest_years: tuple = (2021, 2022, 2023, 2024, 2025)  # toutes les années où le taux du TAL publié est vérifié (2021+)
    panel_entry_year: int = 2022      # immeubles présents avant les mises en service de 2023-2024 : panel fixe
    # --- appariement (valeurs du notebook de départ, justifiées et testées en section 4d) ---
    pair_gap_min_years: float = 0.5   # plus court = probablement le même bail enregistré deux fois
    pair_gap_max_years: float = 2.5   # plus long = bail non comparable
    sensitivity_gap_windows: tuple = ((0.75, 1.5), (11 / 12, 13 / 12))   # fenêtres alternatives : resserrée, et « 12 mois à un mois près »
    asking_gap_min_years: float = 0.25   # deux affichages d'une même unité doivent être espacés d'au moins un trimestre
    days_per_year: float = 365.25
    months_per_year: int = 12
    outlier_tail_share: float = 0.005 # extrémités rognées dans l'analyse de sensibilité (de chaque côté)
    winsor_share: float = 0.01        # winsorisation alternative
    min_cell_size: int = 15           # nombre minimal de paires (ou d'unités) pour publier une cellule seule
    # --- concessions ---
    concession_link_grace_days: int = 30   # une concession peut débuter avant le bail ; valeur qui maximise la concordance (testé en 5b)
    concession_grace_days_tested: tuple = (0, 15, 30, 45, 60, 90)
    term_months_buckets: tuple = (6, 12, 18, 24)   # durées contractuelles réelles ; durées arrondies aux termes contractuels ; distribution vérifiée en section 2
    exact_tolerance_dollars: float = 1.0           # écart d'arrondi accepté entre notre reconstruction et le loyer effectif de Yardi
    min_reconstruction_share: float = 0.85         # part minimale des baux dont rent_effective se reconstruit à partir des concessions
    # --- statistiques ---
    n_bootstrap: int = 1000
    n_repeat_bootstrap: int = 200     # moins de tirages pour l'indice de loyers répétés : chaque tirage refait deux régressions
    n_simulations: int = 5000
    random_seed: int = 2026
    ci_low: float = 10.0              # bornes d'intervalle (percentiles)
    ci_high: float = 90.0
    recent_years_for_mean: int = 3    # référence naïve « moyenne des dernières années »
    min_years_for_reconciliation: int = 4   # moins d'années communes : pas de corrélation rapportée
    mix_sensitivity_ratio: float = 3.0      # le contre-exemple doit être au moins autant de fois plus sensible au mix
    max_method_disagreement_points: float = 1.0   # tolérance de concordance entre estimateurs indépendants (section 9)
    variance_floor: float = 1e-9      # plancher de variance prédite (pondération de Case-Shiller)
    # --- règles publiques (valeurs vérifiées, voir external/SOURCES.md) ---
    ontario_exemption_cutoff: str = "2018-11-15"  # le plafond ne s'applique pas aux unités occupées pour la première fois après cette date
    ontario_min_months_between_increases: int = 12
    month_tolerance_between_increases: int = 1    # tolérance (en mois) : les baux « de 12 mois » durent 11,9 mois
    tal_window_first_month: int = 4   # un taux TAL publié en janvier de t couvre les baux débutant du 2 avril de t au 1er avril de t+1
    tal_window_first_day: int = 2     # ... premier jour de la fenêtre : le 2 avril (un bail débutant le 1er avril relève du taux de l'année précédente)
    new_building_age_years: int = 5   # art. 1955 C.c.Q. : pas de fixation par le Tribunal dans les cinq ans suivant la date où l'immeuble est prêt pour l'usage (si le bail le prévoit)
    # --- contrôles d'intégrité (comptes annoncés par le défi) ---
    expected_leases: int = 4302
    expected_units: int = 1061
    expected_concessions: int = 3560
    expected_asking: int = 3857
    expected_key_collisions: int = 248   # unités en collision sur site + unité (énoncé du défi)
    starter_contract_median_2025: float = 7.04   # médiane contractuelle 2025 du notebook de départ (parité vérifiée en section 4)
    starter_effective_median_2025: float = 3.58  # médiane effective 2025 du notebook de départ
    check_tolerance: float = 0.01                # tolérance des contrôles de parité (en points)
    ask_match_tolerance_days: int = 62           # un affichage est associé à une signature s'il date d'au plus deux mois avant elle
    reference_bedrooms: int = 2                  # segment de référence pour comparer les niveaux à la SCHL (le plus nombreux des deux côtés)
    bedroom_levels: tuple = (0, 1, 2, 3)
    # --- affichage ---
    prose_digits: int = 1             # décimales des nombres rédigés par le code
    table_digits: int = 2             # décimales des tableaux
    fine_digits: int = 3              # décimales des coefficients estimés


@dataclass(frozen=True)
class PlotStyle:
    figsize_wide: tuple = (9.5, 4.5)
    figsize_grid: tuple = (11, 7)
    legend_fontsize: int = 8
    small_fontsize: int = 7
    line_width: float = 2.0
    area_alpha: float = 0.8
    band_alpha: float = 0.15
    error_capsize: int = 5
    x_margin_years: float = 0.5
    marker_jitter_years: float = 0.08            # décalage horizontal pour ne pas superposer deux barres d'erreur
    rc: dict = field(default_factory=lambda: {"figure.dpi": 100, "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False})
    palette: dict = field(default_factory=lambda: {"contract": "#1f77b4", "effective": "#d62728", "naive": "#7f7f7f", "market": "#2ca02c",
                                                   "rule": "#ff7f0e", "alternative": "#9467bd"})


CONFIG = Config()
STYLE = PlotStyle()
PALETTE = STYLE.palette
UNIT_KEY = ["prop_code", "unit_code"]      # clé d'unité : JAMAIS site + unit_code
CONFIG.output_dir.mkdir(parents=True, exist_ok=True)
matplotlib.rcParams.update(STYLE.rc)


def year_start(year):
    """Premier jour de l'année civile."""
    return pd.Period(year, "Y").start_time


def year_end(year):
    """Dernier jour de l'année civile (date de coupure d'une prévision faite à la fin de l'année)."""
    return pd.Period(year, "Y").end_time.normalize()


def fmt(value, digits=CONFIG.prose_digits, signed=False):
    """Nombre à la française pour le texte rédigé par le code (virgule décimale, espace fine des milliers, vrai signe moins)."""
    text = f"{value:+,.{digits}f}" if signed else f"{value:,.{digits}f}"
    return text.replace(",", " ").replace(".", ",").replace("-", "−")


def narrate(text):
    """Affiche un paragraphe Markdown dont tous les nombres viennent de variables calculées (aucun chiffre tapé à la main)."""
    display(Markdown(text))

# %% [markdown]
# ## 1. Définir la question
#
# « La hausse de loyer de l'année » désigne en réalité **cinq nombres différents** : (1) même unité, loyer *contractuel* ; (2) même unité, loyer *effectif* ; (3) renouvellements seulement ; (4) relocations seulement ; (5) portefeuille global (médiane de tous les baux). Ils ne concordent pas, et le choix change la réponse de plusieurs points (section 7).
#
# **Notre choix (nommé et défendu ; la preuve chiffrée est en section 7) :**
#
# | Rôle | Définition | Pourquoi |
# |---|---|---|
# | **Principal** | **Hausse à unité constante du loyer *effectif*** (net des concessions), annualisée, baux débutant en 2026, médiane des paires, renouvellements + relocations pondérés par leur poids réel | Mesure typique du changement de revenu à unité constante ; la moyenne pondérée par le loyer est publiée à côté pour le budget. Elle ne dépend pas du mix du portefeuille. Les concessions deviennent la norme (section 5), donc le loyer contractuel surestime le revenu. |
# | **Secondaire (toujours publié)** | Hausse à unité constante du loyer *contractuel*, renouvellements et relocations séparés | C'est le nombre que le TAL et l'Ontario encadrent : il permet de lire la règle applicable et de calculer les lettres de renouvellement. |
# | **Contre-exemple** | Médiane des loyers de tous les baux, d'une année à l'autre | Mesure surtout la composition du portefeuille (le « pic de 2023 » qui n'a pas eu lieu, section 3). |
#
# **Choix de forme** (chacun est testé en section 7) : année = année de début de bail (c'est l'année budgétaire ; le délai signature → début est de quelques semaines, section 7b) ; annualisation par `(rent / prev_rent) ** (1 / gap) - 1` ; agrégation par médiane (robuste), avec la moyenne pondérée par le loyer en contrôle ; Québec et Ontario **jamais mélangés dans une règle**, seulement dans le total.

# %% [markdown]
# ## 2. Lire les données : ce qu'elles représentent réellement
#
# Les descriptions ne correspondent pas toujours aux données. Cette section (a) charge les quatre fichiers, (b) **vérifie par du code** chaque doute (colonne par colonne), (c) renomme les colonnes selon ce qu'elles *sont*.
#
# **Cette cellule modifie les données** : elle crée quatre tables renommées (`leases`, `units`, `concessions`, `asking`). Convention Yardi : `h*` = identifiant de jointure (`*_id`), `s*` = valeur lue par un humain. On joint sur les identifiants et on regroupe sur les valeurs.

# %%
LEASE_RENAME = {
    "hUnit": "unit_id", "hBuilding": "building_id", "hProperty": "property_id",
    "sPropCode": "prop_code", "sUnitCode": "unit_code", "sBuilding": "building", "sCity": "city",
    "sState": "province", "sSite": "site", "sRent": "rent_contract", "sRentEffective": "rent_effective",
    "sConcession": "has_concession", "sRenewal": "is_renewal", "sTermSeq": "unit_lease_seq",
    "sLeaseFrom": "lease_start", "sLeaseTo": "lease_end", "sSignDate": "sign_date",
    "sTermMonths": "term_months_raw", "sLeaseTerm": "term_label", "sBeds": "beds", "sBaths": "baths",
    "sSqft": "sqft", "sFloor": "floor", "sUnitType": "unit_type", "sAvailable": "lease_start_duplicate",
    "sLatitude": "latitude", "sLongitude": "longitude",
}
CONCESSION_RENAME = {
    "hUnit": "unit_id", "sPropCode": "prop_code", "sUnitCode": "unit_code", "sBuilding": "building",
    "sState": "province", "sBeds": "beds", "sChargeCode": "charge_code", "sChargeDesc": "charge_desc",
    "sAmount": "amount", "sDateFrom": "date_from", "sDateTo": "date_to", "sMonths": "months_covered",
}
ASKING_RENAME = {
    "hUnit": "unit_id", "sPropCode": "prop_code", "sUnitCode": "unit_code", "sBuilding": "building",
    "sState": "province", "sBeds": "beds", "sMonth": "ask_month", "sAskingRent": "rent_ask", "sSqft": "sqft",
}
UNIT_RENAME = {
    "hUnit": "unit_id", "sPropCode": "prop_code", "sUnitCode": "unit_code", "sBuilding": "building",
    "sState": "province", "sSite": "site", "sBeds": "beds", "sSqft": "sqft", "sRent": "rent_listed_2025",
    "sAsOf": "listed_as_of_month", "sAvailable": "available_date", "sFloor": "floor",
}


def load_table(file_name, rename_map, date_columns=()):
    """Lit un CSV source (jamais modifié) et retourne une copie renommée, colonnes de dates converties."""
    table = pd.read_csv(CONFIG.data_dir / file_name)
    for column in date_columns:
        table[column] = pd.to_datetime(table[column])
    return table.rename(columns=rename_map)


leases = load_table("equinoxe_lease_history.csv", LEASE_RENAME, ["sLeaseFrom", "sLeaseTo", "sSignDate", "sAvailable"])
units = load_table("equinoxe_listings.csv", UNIT_RENAME, ["sAvailable"])
concessions = load_table("equinoxe_concessions.csv", CONCESSION_RENAME, ["sDateFrom", "sDateTo"])
asking = load_table("equinoxe_asking_history.csv", ASKING_RENAME)
asking["ask_date"] = pd.PeriodIndex(asking.ask_month, freq="M").start_time

for name, table in [("leases", leases), ("units", units), ("concessions", concessions), ("asking", asking)]:
    print(f"{name:<12} {len(table):>6,} lignes   {table.shape[1]:>2} colonnes")

# %% [markdown]
# ### 2b. Audit « description contre réalité »
# Chaque ligne du tableau ci-dessous est une *vérification exécutée* (pas une affirmation). Les doutes viennent de l'énoncé : « la description ne correspond pas toujours aux données ».
#
# **Cette cellule ajoute des colonnes** à `leases` : `lease_id` (identifiant technique), `lease_year` (année de début), `drag` (part du loyer cédée en concession), `rank_in_unit` et `term_months` (durée arrondie à la durée contractuelle réelle).

# %%
leases = leases.sort_values(UNIT_KEY + ["lease_start"]).reset_index(drop=True)
leases["lease_id"] = leases.index                       # identifiant technique de bail (pour les jointures)
leases["lease_year"] = leases.lease_start.dt.year       # année budgétaire = année de début de bail
leases["drag"] = 1 - leases.rent_effective / leases.rent_contract          # part du loyer cédée en concession
leases["rank_in_unit"] = leases.groupby(UNIT_KEY).cumcount() + 1
leases["term_months"] = leases.term_months_raw.apply(
    lambda raw: min(CONFIG.term_months_buckets, key=lambda bucket: abs(bucket - raw))).astype(int)   # 5,9 -> 6 ; 11,9 -> 12 ; ...

key_collisions = units.duplicated(["site", "unit_code"], keep=False).sum()
unit_id_is_key = (leases.groupby(UNIT_KEY).unit_id.nunique() == 1).all() and (leases.groupby("unit_id").size().index.size == leases.groupby(UNIT_KEY).ngroups)


def link_concession_lines(concession_table, lease_table, grace_days):
    """Rattache chaque ligne de concession au dernier bail de la même unité commencé au plus tard `grace_days` jours après le début de la ligne."""
    keyed = concession_table.assign(link_date=concession_table.date_from + pd.Timedelta(days=grace_days)).sort_values("link_date")
    lease_columns = UNIT_KEY + ["lease_id", "lease_start", "lease_end", "term_months", "rent_contract", "rent_effective"]
    attached = pd.merge_asof(keyed, lease_table[lease_columns].sort_values("lease_start"), left_on="link_date", right_on="lease_start", by=UNIT_KEY, direction="backward")
    attached = attached[attached.lease_id.notna() & (attached.date_from <= attached.lease_end)].copy()
    attached["monthly_equivalent"] = -attached.amount * attached.months_covered / attached.term_months
    return attached


linked_for_audit = link_concession_lines(concessions, leases, CONFIG.concession_link_grace_days)
promo_lines_audit = linked_for_audit[linked_for_audit.charge_code == "PromoPay"]
promo_ratio = (-promo_lines_audit.amount / promo_lines_audit.rent_contract).median()
last_lease = leases.groupby(UNIT_KEY).tail(1)[UNIT_KEY + ["rent_contract"]]
listing_gap = (units.merge(last_lease, on=UNIT_KEY).eval("rent_listed_2025 / rent_contract - 1") * 100)
labels_with_several_terms = int((leases.groupby("term_label").term_months.nunique() > 1).sum())

audit = pd.DataFrame([
    ("sAvailable (baux)", "date de disponibilité", f"identique à `lease_start` sur {(leases.lease_start_duplicate == leases.lease_start).mean():.0%} des baux",
     "redondante : renommée `lease_start_duplicate`, non utilisée"),
    ("sConcession", "indicateur de concession", f"égal à `rent_effective < rent_contract` sur {((leases.rent_effective < leases.rent_contract) == leases.has_concession.astype(bool)).mean():.0%} des baux",
     "indicateur dérivé : on utilise plutôt `drag` (la profondeur)"),
    ("sTermSeq", "numéro de terme", f"rang du bail dans l'unité : {(leases.unit_lease_seq == leases.rank_in_unit).mean():.0%} concordent ; {int(((leases.unit_lease_seq >= 2) & (leases.is_renewal == 0)).sum()):,} baux de rang ≥ 2 sont des relocations",
     "ce n'est PAS un compteur de renouvellements : on sépare avec `is_renewal`"),
    ("sTermMonths", "durée en mois", f"valeurs {[float(v) for v in sorted(leases.term_months_raw.unique())]} : des jours/30,4 arrondis",
     f"arrondie à {'/'.join(str(bucket) for bucket in CONFIG.term_months_buckets)} mois → `term_months`"),
    ("sLeaseTerm", "libellé de durée", f"incohérent avec la durée réelle : {labels_with_several_terms} libellé(s) regroupent plusieurs durées réelles",
     "ignorée au profit de `term_months`"),
    ("PromoPay", "frais mensuel (« payable in the future »)", f"`months_covered` ∈ {[int(v) for v in sorted(concessions[concessions.charge_code == 'PromoPay'].months_covered.unique())]}, montant ≈ {promo_ratio:.2f} × le loyer mensuel",
     "montant **forfaitaire unique** (≈ 1 mois), étalé sur la durée du bail (section 5)"),
    ("sRent (listings)", "loyer", f"loyer *affiché* de 2025 (écart médian de {listing_gap.median():+.1f} % contre le dernier bail, plage {listing_gap.min():.0f} à {listing_gap.max():.1f} %)",
     "renommé `rent_listed_2025` ; ce n'est pas un bail (et pas un prix 2026, voir section 10g)"),
    ("sSite + sUnitCode", "identifiant d'unité ?", f"{key_collisions} unités sur {len(units):,} entrent en collision",
     "jamais une clé : on utilise `prop_code` + `unit_code`"),
    ("hUnit ↔ prop_code+unit_code", "clés équivalentes ?", f"correspondance 1 pour 1 : {bool(unit_id_is_key)}", "les deux clés sont sûres"),
    ("dates de fin / concessions", "historique jusqu'à 2025", f"`lease_end` jusqu'en {leases.lease_end.max():%Y-%m} ; concessions jusqu'en {concessions.date_to.max():%Y-%m}",
     "dates *contractuelles* futures : elles disent quelles unités échoient en 2026 (cohorte)"),
], columns=["colonne source", "description officielle", "ce que le code constate", "décision"])
audit

# %% [markdown]
# **À retenir de l'audit.** Trois pièges changent des résultats : `PromoPay` (forfait unique, pas mensuel), `sTermSeq` (rang du bail, pas renouvellement) et `sSite` (jamais une clé). `sAvailable` est un doublon. Nous ne modifions aucun fichier source.

# %%
# Contrôles de validité (tests exécutés à chaque lancement) -------------------------------------------------
assert (len(leases), len(units), len(concessions), len(asking)) == (CONFIG.expected_leases, CONFIG.expected_units, CONFIG.expected_concessions, CONFIG.expected_asking), "comptes de lignes inattendus"
assert (leases.lease_start_duplicate == leases.lease_start).all()
assert ((leases.rent_effective < leases.rent_contract) == leases.has_concession.astype(bool)).all()
assert key_collisions == CONFIG.expected_key_collisions, f"le défi annonce {CONFIG.expected_key_collisions} collisions sur site + unité"
assert leases.groupby(UNIT_KEY).unit_id.nunique().eq(1).all()
print("Contrôles de la section 2 : OK")
