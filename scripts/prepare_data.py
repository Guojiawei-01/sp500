"""Prepare headline, macro, and EDA summary datasets for the capstone project."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_HEADLINES = PROJECT_ROOT / "data" / "raw" / "sp500_headlines_2008_2024.csv"
RAW_MACRO_DIR = PROJECT_ROOT / "data" / "raw" / "macro"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "report"


@dataclass(frozen=True)
class FredSeries:
    series_id: str
    column: str
    description: str


FRED_SERIES = [
    FredSeries("VIXCLS", "vix", "CBOE Volatility Index"),
    FredSeries("DGS10", "treasury_10y", "10-year Treasury constant maturity rate"),
    FredSeries("DGS2", "treasury_2y", "2-year Treasury constant maturity rate"),
    FredSeries("FEDFUNDS", "fed_funds_rate", "Federal funds effective rate"),
    FredSeries("CPIAUCSL", "cpi_all_items", "CPI for all urban consumers"),
    FredSeries("UNRATE", "unemployment_rate", "Unemployment rate"),
    FredSeries("USREC", "recession_indicator", "NBER recession indicator"),
]


def ensure_dirs() -> None:
    RAW_MACRO_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


def load_headlines() -> pd.DataFrame:
    df = pd.read_csv(RAW_HEADLINES, parse_dates=["Date"])
    expected = {"Title", "Date", "CP"}
    missing = expected.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.copy()
    df["Title"] = df["Title"].astype(str).str.strip()
    df["CP"] = pd.to_numeric(df["CP"], errors="coerce")
    df = df.dropna(subset=["Title", "Date", "CP"])
    df = df[df["Title"].str.len() > 0]
    return df.sort_values(["Date", "Title"]).reset_index(drop=True)


def clean_headlines(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    clean = df.drop_duplicates().sort_values(["Date", "Title"]).reset_index(drop=True)

    daily = (
        clean.groupby("Date")
        .agg(
            CP=("CP", "first"),
            headline_count=("Title", "size"),
            unique_headline_count=("Title", "nunique"),
            daily_text=("Title", lambda values: " . ".join(values.astype(str))),
        )
        .reset_index()
        .sort_values("Date")
    )
    daily["close_next_day"] = daily["CP"].shift(-1)
    daily["return_next_day"] = daily["close_next_day"] / daily["CP"] - 1
    daily["direction_next_day"] = (daily["return_next_day"] > 0).astype(int)
    daily["return_same_day"] = daily["CP"].pct_change()
    daily = daily.dropna(subset=["return_next_day"]).reset_index(drop=True)
    daily["year"] = daily["Date"].dt.year
    daily["month"] = daily["Date"].dt.to_period("M").astype(str)
    daily["regime"] = daily["Date"].apply(assign_regime)
    return clean, daily


def assign_regime(date: pd.Timestamp) -> str:
    if pd.Timestamp("2008-01-01") <= date <= pd.Timestamp("2009-12-31"):
        return "2008-2009 financial crisis"
    if pd.Timestamp("2020-01-01") <= date <= pd.Timestamp("2020-12-31"):
        return "2020 COVID shock"
    if pd.Timestamp("2022-01-01") <= date <= pd.Timestamp("2023-12-31"):
        return "2022-2023 rate-hike cycle"
    return "Other"


def fred_url(series_id: str) -> str:
    return f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"


def fetch_fred_series(series: FredSeries) -> pd.DataFrame:
    df = pd.read_csv(fred_url(series.series_id), na_values=".")
    df = df.rename(columns={"observation_date": "Date", "DATE": "Date", series.series_id: series.column})
    if "Date" not in df.columns or series.column not in df.columns:
        raise ValueError(f"Unexpected FRED response for {series.series_id}: {df.columns.tolist()}")
    df["Date"] = pd.to_datetime(df["Date"])
    df[series.column] = pd.to_numeric(df[series.column], errors="coerce")
    df = df[["Date", series.column]].sort_values("Date")
    df.to_csv(RAW_MACRO_DIR / f"{series.series_id}.csv", index=False)
    return df


def fetch_macro_data() -> pd.DataFrame:
    frames = []
    for series in FRED_SERIES:
        frame = fetch_fred_series(series)
        if series.column == "cpi_all_items":
            frame["cpi_yoy"] = frame["cpi_all_items"].pct_change(periods=12, fill_method=None)
        frames.append(frame)

    macro = frames[0]
    for frame in frames[1:]:
        macro = macro.merge(frame, on="Date", how="outer")
    macro = macro.sort_values("Date").reset_index(drop=True)

    if {"treasury_10y", "treasury_2y"}.issubset(macro.columns):
        macro["term_spread_10y_2y"] = macro["treasury_10y"] - macro["treasury_2y"]

    macro.to_csv(RAW_MACRO_DIR / "fred_macro_combined.csv", index=False)
    return macro


def asof_join_macro(daily: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    out = daily.sort_values("Date").copy()
    macro = macro.sort_values("Date").copy()
    macro_cols = [col for col in macro.columns if col != "Date"]

    for col in macro_cols:
        series = macro[["Date", col]].dropna().sort_values("Date")
        out = pd.merge_asof(out, series, on="Date", direction="backward")

    return out


def write_summaries(raw: pd.DataFrame, clean: pd.DataFrame, daily: pd.DataFrame, daily_macro: pd.DataFrame) -> None:
    cp_values_per_date = raw.groupby("Date")["CP"].nunique()
    macro_cols = [
        "vix",
        "treasury_10y",
        "treasury_2y",
        "term_spread_10y_2y",
        "fed_funds_rate",
        "cpi_all_items",
        "cpi_yoy",
        "unemployment_rate",
        "recession_indicator",
    ]
    macro_coverage = (
        daily_macro[macro_cols]
        .notna()
        .mean()
        .rename("coverage_rate")
        .reset_index()
        .rename(columns={"index": "field"})
    )

    headlines_by_year = (
        clean.assign(year=clean["Date"].dt.year)
        .groupby("year")
        .agg(
            headline_rows=("Title", "size"),
            trading_days=("Date", "nunique"),
            avg_headlines_per_day=("Title", lambda x: len(x) / clean.loc[x.index, "Date"].nunique()),
        )
        .reset_index()
    )

    regime_summary = (
        daily_macro.groupby("regime")
        .agg(
            days=("Date", "count"),
            avg_headlines=("headline_count", "mean"),
            avg_next_return=("return_next_day", "mean"),
            volatility=("return_next_day", "std"),
            up_day_rate=("direction_next_day", "mean"),
            avg_vix=("vix", "mean"),
            avg_fed_funds=("fed_funds_rate", "mean"),
            avg_10y_yield=("treasury_10y", "mean"),
        )
        .reset_index()
    )

    summary = {
        "raw_rows": int(len(raw)),
        "clean_rows": int(len(clean)),
        "exact_duplicate_rows_removed": int(raw.duplicated().sum()),
        "date_min": str(raw["Date"].min().date()),
        "date_max": str(raw["Date"].max().date()),
        "unique_trading_dates_raw": int(raw["Date"].nunique()),
        "unique_trading_dates_modeling": int(daily_macro["Date"].nunique()),
        "dates_with_multiple_cp_values": int((cp_values_per_date > 1).sum()),
        "missing_values_raw": {k: int(v) for k, v in raw.isna().sum().to_dict().items()},
        "headline_count_per_day": {
            "mean": float(daily["headline_count"].mean()),
            "median": float(daily["headline_count"].median()),
            "max": int(daily["headline_count"].max()),
        },
        "next_day_return": {
            "mean": float(daily["return_next_day"].mean()),
            "std": float(daily["return_next_day"].std()),
            "min": float(daily["return_next_day"].min()),
            "max": float(daily["return_next_day"].max()),
            "up_day_rate": float(daily["direction_next_day"].mean()),
        },
        "macro_coverage": {
            row["field"]: float(row["coverage_rate"]) for _, row in macro_coverage.iterrows()
        },
    }

    headlines_by_year.to_csv(PROCESSED_DIR / "eda_headlines_by_year.csv", index=False)
    regime_summary.to_csv(PROCESSED_DIR / "eda_regime_summary.csv", index=False)
    macro_coverage.to_csv(PROCESSED_DIR / "eda_macro_coverage.csv", index=False)
    (PROCESSED_DIR / "eda_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (REPORT_DIR / "data_preparation_summary.md").write_text(summary_markdown(summary), encoding="utf-8")


def summary_markdown(summary: dict) -> str:
    macro_lines = "\n".join(
        f"- `{field}`: {coverage:.1%} coverage"
        for field, coverage in summary["macro_coverage"].items()
    )
    return f"""# Data Preparation Summary

## Raw Data Quality

- Raw rows: {summary["raw_rows"]:,}
- Clean rows after exact duplicate removal: {summary["clean_rows"]:,}
- Exact duplicate rows removed: {summary["exact_duplicate_rows_removed"]:,}
- Date range: {summary["date_min"]} to {summary["date_max"]}
- Unique trading dates in raw data: {summary["unique_trading_dates_raw"]:,}
- Dates with multiple S&P 500 close values: {summary["dates_with_multiple_cp_values"]}
- Missing values in raw data: {summary["missing_values_raw"]}

## Daily Modeling Table

- Unique trading dates after target creation: {summary["unique_trading_dates_modeling"]:,}
- Average headlines per trading day: {summary["headline_count_per_day"]["mean"]:.2f}
- Median headlines per trading day: {summary["headline_count_per_day"]["median"]:.0f}
- Maximum headlines on one trading day: {summary["headline_count_per_day"]["max"]:,}
- Mean next-day return: {summary["next_day_return"]["mean"]:.4%}
- Next-day return standard deviation: {summary["next_day_return"]["std"]:.4%}
- Up-day rate: {summary["next_day_return"]["up_day_rate"]:.1%}

## Macro Coverage

{macro_lines}

## Notes for Modeling

- Exact duplicate rows are removed before daily aggregation.
- The target is next trading day's return, not same-day return.
- FRED macro variables are joined by most recent available observation date using an as-of join.
- Monthly macro variables may not perfectly reflect publication timing; treat them as controls unless release lags are modeled explicitly.
"""


def main() -> None:
    ensure_dirs()
    raw = load_headlines()
    clean, daily = clean_headlines(raw)
    macro = fetch_macro_data()
    daily_macro = asof_join_macro(daily, macro)

    clean.to_csv(PROCESSED_DIR / "headlines_clean.csv", index=False)
    daily.to_csv(PROCESSED_DIR / "daily_dataset.csv", index=False)
    daily_macro.to_csv(PROCESSED_DIR / "daily_with_macro.csv", index=False)
    write_summaries(raw, clean, daily, daily_macro)

    print("Data preparation complete.")
    print(f"Clean headlines: {PROCESSED_DIR / 'headlines_clean.csv'}")
    print(f"Daily dataset: {PROCESSED_DIR / 'daily_dataset.csv'}")
    print(f"Daily with macro: {PROCESSED_DIR / 'daily_with_macro.csv'}")
    print(f"Summary: {REPORT_DIR / 'data_preparation_summary.md'}")


if __name__ == "__main__":
    main()
