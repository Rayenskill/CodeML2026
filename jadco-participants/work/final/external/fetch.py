"""Build the tidy public-data tables used by the notebook.

Sources (all public, see SOURCES.md for licences):
  * Statistics Canada table 18-10-0004-01 (CPI, monthly): rent component, Quebec / Ontario / Canada
  * Statistics Canada tables 34-10-0130-01 and 34-10-0133-01 (CMHC rental market survey,
    vacancy rate and average rent by bedrooms), Montreal CMA and Ottawa-Gatineau (Ontario part)
  * Hand-entered official rates (TAL, Ontario guideline, CMHC headline figures): manual_public_rates.csv

Every output row carries `published_on`, the first date at which the value was public.
The backtest uses it to keep only what was known at the forecast date (no leakage).

Run:  python external/fetch.py   (downloads the raw StatCan CSV zips if missing)
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"
TIDY = HERE / "tidy"

STATCAN_CSV_URL = "https://www150.statcan.gc.ca/n1/tbl/csv/{table}-eng.zip"
STATCAN_TABLES = ("18100004", "34100130", "34100133")
RETRIEVED_ON = "2026-10-04"

CPI_PUBLICATION_LAG_DAYS = 20          # StatCan publishes the CPI around the 20th of the following month
CMHC_PUBLICATION_MONTH_DAY = (12, 15)  # CMHC publishes the October survey in mid-December
CPI_GEOS = {"Quebec": "Quebec", "Ontario": "Ontario", "Canada": "Canada"}
CMHC_GEOS = {
    "Montréal, Quebec": "Montreal CMA",
    "Ottawa-Gatineau, Ontario part, Ontario/Quebec": "Ottawa CMA (Ontario part)",
}
CMHC_STRUCTURE = "Row and apartment structures of three units and over"
BEDROOM_LABELS = {
    "Bachelor units": 0, "One bedroom units": 1, "Two bedroom units": 2, "Three bedroom units": 3,
}
MONTHS_PER_YEAR = 12
CPI_URL = "https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=1810000401"
CMHC_VACANCY_URL = "https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=3410013001"
CMHC_RENT_URL = "https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=3410013301"


def download_raw() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    for table in STATCAN_TABLES:
        if (RAW / f"{table}.csv").exists():
            continue
        response = requests.get(STATCAN_CSV_URL.format(table=table), timeout=180)
        response.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            archive.extractall(RAW)


def build_cpi_rent() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Monthly CPI rent index and the annual summaries (annual-average and December/December)."""
    cpi = pd.read_csv(RAW / "18100004.csv", low_memory=False)
    cpi = cpi[(cpi["Products and product groups"] == "Rent") & cpi.GEO.isin(CPI_GEOS)]
    monthly = cpi.pivot(index="REF_DATE", columns="GEO", values="VALUE")
    monthly.index = pd.to_datetime(monthly.index)
    monthly = monthly.sort_index()
    yoy = (monthly / monthly.shift(MONTHS_PER_YEAR) - 1) * 100

    monthly_rows = []
    for geo, label in CPI_GEOS.items():
        monthly_rows.append(pd.DataFrame({
            "series": "cpi_rent_index", "geo": label, "month": monthly.index,
            "value": monthly[geo].to_numpy(), "yoy_pct": yoy[geo].to_numpy(),
        }))
    monthly_long = pd.concat(monthly_rows, ignore_index=True)
    monthly_long["published_on"] = (monthly_long.month + pd.offsets.MonthBegin(1)
                                    + pd.Timedelta(days=CPI_PUBLICATION_LAG_DAYS)).dt.date
    monthly_long["source_url"] = CPI_URL
    monthly_long["retrieved_on"] = RETRIEVED_ON
    monthly_long["month"] = monthly_long.month.dt.date

    annual_average = monthly.groupby(monthly.index.year).mean()
    december = monthly[monthly.index.month == MONTHS_PER_YEAR].copy()
    december.index = december.index.year
    months_per_year = monthly.groupby(monthly.index.year).size()
    complete_years = months_per_year[months_per_year == MONTHS_PER_YEAR].index
    summaries = {
        "cpi_rent_avg_yoy": (annual_average.pct_change() * 100).loc[complete_years],
        "cpi_rent_dec_yoy": december.pct_change() * 100,
    }
    all_items = pd.read_csv(RAW / "18100004.csv", low_memory=False)
    all_items = all_items[(all_items["Products and product groups"] == "All-items") & (all_items.GEO == "Quebec")]
    all_items_monthly = all_items.set_index(pd.to_datetime(all_items.REF_DATE)).VALUE.sort_index()
    all_items_yoy = (all_items_monthly / all_items_monthly.shift(MONTHS_PER_YEAR) - 1) * 100
    all_items_annual = all_items_monthly.groupby(all_items_monthly.index.year).mean().pct_change() * 100
    all_items_ytd = all_items_yoy.groupby(all_items_yoy.index.year).mean()          # moyenne des variations mensuelles des mois publiés
    annual_rows = []
    for year, value in all_items_annual.dropna().items():
        if year in complete_years:
            annual_rows.append({"series": "cpi_all_items_avg_yoy", "geo": "Quebec", "year": int(year), "value": round(float(value), 3), "unit": "percent",
                                "published_on": pd.Timestamp(int(year) + 1, 1, CPI_PUBLICATION_LAG_DAYS).date(), "source_url": CPI_URL,
                                "retrieved_on": RETRIEVED_ON, "confidence": "high", "note": "IPC ensemble, Québec : base de la formule du TAL dès 2026"})
    last_year = int(all_items_yoy.index.max().year)
    annual_rows.append({"series": "cpi_all_items_ytd_yoy", "geo": "Quebec", "year": last_year, "value": round(float(all_items_ytd[last_year]), 3), "unit": "percent",
                        "published_on": all_items_yoy.index.max().date(), "source_url": CPI_URL, "retrieved_on": RETRIEVED_ON, "confidence": "high",
                        "note": f"Moyenne des variations annuelles, janvier à {all_items_yoy.index.max():%B %Y} (année en cours)"})
    for series, table in summaries.items():
        for geo, label in CPI_GEOS.items():
            for year, value in table[geo].dropna().items():
                annual_rows.append({
                    "series": series, "geo": label, "year": int(year), "value": round(float(value), 3),
                    "unit": "percent",
                    "published_on": pd.Timestamp(int(year) + 1, 1, CPI_PUBLICATION_LAG_DAYS).date(),
                    "source_url": CPI_URL, "retrieved_on": RETRIEVED_ON, "confidence": "high",
                    "note": "Computed from table 18-10-0004-01",
                })
    return monthly_long, pd.DataFrame(annual_rows)


def build_cmhc() -> pd.DataFrame:
    """CMHC October survey via StatCan: vacancy rate and average rent (level and growth) by bedrooms."""
    rows = []
    vacancy = pd.read_csv(RAW / "34100130.csv", low_memory=False)
    vacancy = vacancy[vacancy.GEO.isin(CMHC_GEOS)]
    for record in vacancy.itertuples():
        rows.append(dict(series="cmhc_vacancy_rate", geo=CMHC_GEOS[record.GEO], year=int(record.REF_DATE),
                         value=float(record.VALUE), unit="percent", source_url=CMHC_VACANCY_URL))
    rents = pd.read_csv(RAW / "34100133.csv", low_memory=False)
    rents = rents[rents.GEO.isin(CMHC_GEOS) & (rents["Type of structure"] == CMHC_STRUCTURE)]
    for (geo, unit_type), group in rents.groupby(["GEO", "Type of unit"]):
        series = group.set_index("REF_DATE").VALUE.sort_index()
        beds = BEDROOM_LABELS[unit_type]
        growth = series.pct_change() * 100
        for year in series.index:
            rows.append(dict(series=f"cmhc_avg_rent_beds{beds}", geo=CMHC_GEOS[geo], year=int(year),
                             value=float(series[year]), unit="dollars", source_url=CMHC_RENT_URL))
            if pd.notna(growth[year]):
                rows.append(dict(series=f"cmhc_avg_rent_growth_beds{beds}", geo=CMHC_GEOS[geo], year=int(year),
                                 value=round(float(growth[year]), 3), unit="percent", source_url=CMHC_RENT_URL))
    frame = pd.DataFrame(rows)
    month, day = CMHC_PUBLICATION_MONTH_DAY
    frame["published_on"] = [pd.Timestamp(year, month, day).date() for year in frame.year]
    frame["retrieved_on"] = RETRIEVED_ON
    frame["confidence"] = "high"
    frame["note"] = "CMHC October survey, republished by Statistics Canada"
    return frame


def main() -> None:
    download_raw()
    TIDY.mkdir(parents=True, exist_ok=True)
    cpi_monthly, cpi_annual = build_cpi_rent()
    cmhc = build_cmhc()
    manual = pd.read_csv(HERE / "manual_public_rates.csv", parse_dates=["published_on"])
    manual["published_on"] = manual.published_on.dt.date
    columns = ["series", "geo", "year", "value", "unit", "published_on", "source_url", "retrieved_on",
               "confidence", "note"]
    annual = pd.concat([cpi_annual, cmhc, manual], ignore_index=True)[columns]
    annual = annual[annual.year >= 2015].sort_values(["series", "geo", "year"]).reset_index(drop=True)
    annual.to_csv(TIDY / "external_annual.csv", index=False)
    cpi_monthly.to_csv(TIDY / "cpi_rent_monthly.csv", index=False)
    print(f"external_annual.csv: {len(annual)} rows, {annual.series.nunique()} series")
    print(f"cpi_rent_monthly.csv: {len(cpi_monthly)} rows")


if __name__ == "__main__":
    main()
