"""Shared feature construction for every EquiAlgo strategy."""
import numpy as np
import pandas as pd

REMOTE = ["Bas-Saint-Laurent", "Cote-Nord", "Gaspesie-Iles-de-la-Madeleine"]


def features(df, region_mode="remote_flag", include_postal=False):
    X = pd.DataFrame(index=df.index)
    X["cote_r"] = df.cote_r_equivalent
    X["log_rev"] = np.log(df.revenu_familial_estime)
    X["rev"] = df.revenu_familial_estime / 1e5
    X["heures"] = df.heures_travail_semaine
    X["log_dist"] = np.log1p(df.distance_domicile_campus_km)
    X["dist"] = df.distance_domicile_campus_km / 100
    X["premgen"] = df.premiere_generation_universitaire
    X = X.join(pd.get_dummies(df.programme_etudes, prefix="prog", drop_first=True).astype(float))
    if region_mode == "remote_flag":
        X["remote"] = df.region_administrative.isin(REMOTE).astype(float)
    else:  # five regions, Montreal as reference
        X = X.join(pd.get_dummies(df.region_administrative, prefix="reg").drop(columns="reg_Montreal").astype(float))
    if include_postal:
        X = X.join(pd.get_dummies(df.code_postal_3, prefix="cp", drop_first=True).astype(float))
    return X
