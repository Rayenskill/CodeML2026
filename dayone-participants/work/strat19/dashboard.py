"""Strategy 19: privacy-preserving epidemiology aggregates (descriptive only, no risk prediction).

Input: a de-identified analytics view (here `maternal_registry_synthetic.csv`; in production the validated
records without codes / dates finer than month). Output: indicator table with numerators, denominators and
*missing counts* (unknown != negative), small-cell suppression (k=5), age bands, optional Laplace noise.

python dashboard.py --out ~/dayone_local/dashboard.html
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

W = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(W / "shared"), str(W / "strat5")]
from common import DATA  # noqa: E402

K = 5


def load():
    df = pd.read_csv(DATA / "maternal_registry_synthetic.csv")
    df.columns = [c.split(" (")[0].strip() for c in df.columns]
    return df.drop(columns=["id"])


def age_band(a):
    if pd.isna(a):
        return "inconnu"
    return "<20" if a < 20 else "20-34" if a < 35 else "35+"


def indicator(name, num, den, missing, eps=None, rng=None):
    if eps and rng is not None:
        num = max(0, num + rng.laplace(0, 1 / eps)); den = max(1, den + rng.laplace(0, 1 / eps))
    supp = den < K or (0 < num < K)
    return dict(indicator=name, numerator=None if supp else int(round(num)), denominator=None if den < K else int(round(den)),
                rate=None if supp else round(num / den, 3), missing=int(missing),
                note="supprimé (n<5)" if supp else "")


def indicators(df, eps=None, seed=0):
    rng = np.random.default_rng(seed)
    out = []

    def binary(col, name, pos=1):
        s = df[col]
        out.append(indicator(name, int((s == pos).sum()), int(s.notna().sum()), int(s.isna().sum()), eps, rng))
    binary("hiv test result", "VIH positif")
    binary("syphilis test result", "Syphilis positive")
    binary("hepatitis c test result", "Hépatite C positive")
    s = df["mean systolic bp"]; d = df["mean diastolic bp"]
    m = s.notna() & d.notna()
    out.append(indicator("TA >= 140/90", int(((s >= 140) | (d >= 90))[m].sum()), int(m.sum()), int((~m).sum()), eps, rng))
    h = df["hemoglobin"]
    out.append(indicator("Hb < 11 g/dL (anémie)", int((h < 11).sum()), int(h.notna().sum()), int(h.isna().sum()), eps, rng))
    w = df["child birth weight"]
    out.append(indicator("Poids naissance < 2500 g", int((w < 2500).sum()), int(w.notna().sum()), int(w.isna().sum()), eps, rng))
    binary("preterm birth", "Naissance prématurée")
    binary("type of delivery", "Césarienne")
    binary("breastfeeding initiated", "Allaitement initié")
    return pd.DataFrame(out)


def by_age(df):
    g = df.assign(band=df["age"].map(age_band)).groupby("band")
    t = g.agg(n=("age", "size"), hb_mean=("hemoglobin", "mean"), sbp_mean=("mean systolic bp", "mean"),
              hb_missing=("hemoglobin", lambda x: int(x.isna().sum())))
    t.loc[t["n"] < K, ["hb_mean", "sbp_mean"]] = np.nan          # small-cell suppression
    return t.round(1)


def html(ind, ages):
    return f"""<!doctype html><meta charset="utf-8"><title>Agrégats anonymisés</title>
<style>body{{font-family:system-ui;margin:24px;background:#fff;color:#222}}table{{border-collapse:collapse}}
td,th{{border:1px solid #ccc;padding:4px 8px;text-align:right}}th{{background:#f4e1e6}}</style>
<h1>Agrégats anonymisés (données synthétiques)</h1>
<p>Descriptif uniquement, aucune prédiction. Cellules &lt; {K} supprimées ; les valeurs manquantes sont comptées à part
(« inconnu » n'est pas « négatif »).</p>{ind.to_html(index=False)}<h2>Par tranche d'âge</h2>{ages.to_html()}"""


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out"); ap.add_argument("--eps", type=float, default=None)
    a = ap.parse_args()
    df = load()
    ind, ages = indicators(df, a.eps), by_age(df)
    print(ind.to_string(index=False)); print(ages)
    if a.out:
        Path(a.out).write_text(html(ind, ages), encoding="utf-8")


# ---------------------------------------------------------------- from the digitised records (edge box central store)
def _num(v):
    import re
    m = re.match(r"\s*(\d+(?:[.,]\d+)?)", str(v or ""))
    return float(m.group(1).replace(",", ".")) if m else None


def records_view(rows):
    """De-identified analytics view: one row per synced page, only clinical values (no code, no dates)."""
    from collections import defaultdict
    from vocab import canonical
    out = defaultdict(list)
    for r in rows:
        for p in r.get("pages", []):
            for f in p.get("fields", []):
                k, v, st = f["key"], f.get("value"), f.get("status")
                if k.endswith(".ta") or ".ta." in k:
                    out["ta"].append((v, st))
                elif k in ("inline.t", "inline.temperature"):
                    out["temp"].append((_num(v), st))
                elif ".serologie_vih." in k:
                    out["hiv"].append((canonical(v), st))
                elif ".syphilis_tpha_vdrl." in k:
                    out["syph"].append((canonical(v), st))
                elif ".ag_hbs." in k:
                    out["hbs"].append((canonical(v), st))
    return out


def indicators_from_records(rows):
    v = records_view(rows)
    res = []

    def ind(name, vals, pos):
        known = [x for x, st in vals if x is not None]
        missing = sum(1 for x, st in vals if x is None)
        res.append(indicator(name, sum(1 for x in known if pos(x)), len(known), missing))
    import re
    def high_bp(x):
        m = re.match(r"(\d+)/(\d+)", str(x)); return bool(m) and (int(m.group(1)) >= 140 or int(m.group(2)) >= 90)
    ind("TA >= 140/90 (mesures)", v["ta"], high_bp)
    ind("Température >= 38 °C", v["temp"], lambda x: x >= 38)
    ind("VIH positif", v["hiv"], lambda x: x == "POS")
    ind("Syphilis positive", v["syph"], lambda x: x == "POS")
    ind("Ag HBs positif", v["hbs"], lambda x: x == "POS")
    return pd.DataFrame(res)
