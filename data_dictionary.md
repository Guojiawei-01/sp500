# Data Dictionary

Dataset: **S&P 500 with Financial News Headlines (2008-2024)**

Source: Kaggle dataset by dyutidasmahaptra, licensed under CC BY-SA 4.0.

File: `data/raw/sp500_headlines_2008_2024.csv`

## Raw Columns

| Column | Type | Description | Notes |
| --- | --- | --- | --- |
| `Title` | string | Financial news headline text. | Multiple headlines can appear on the same trading date. Some headlines may be duplicated or off-topic. |
| `Date` | date | Trading date associated with the headline and S&P 500 close. | Format is `YYYY-MM-DD`. The current file covers 2008-01-02 to 2024-03-04. |
| `CP` | float | S&P 500 closing price for the date. | `CP` is constant within each date in the current file. |

## Derived Fields Used in Analysis

| Field | Type | Description |
| --- | --- | --- |
| `headline_count` | integer | Number of headlines on a trading date. |
| `unique_headline_count` | integer | Number of unique headlines on a trading date after duplicate checking. |
| `daily_text` | string | Headlines for the same date combined into one text field. |
| `return_next_day` | float | Next trading day's closing price divided by current closing price minus 1. |
| `direction_next_day` | integer | Binary target equal to 1 when `return_next_day` is positive and 0 otherwise. |
| `sentiment_compound` | float | Baseline VADER compound sentiment score. |
| `sentiment_pos` | float | Baseline VADER positive sentiment score. |
| `sentiment_neu` | float | Baseline VADER neutral sentiment score. |
| `sentiment_neg` | float | Baseline VADER negative sentiment score. |
| `regime` | string | Market period label used for segmented analysis. |

## Macro Fields

Macro fields are downloaded from FRED and joined to the daily headline table by most recent available observation date.

| Field | FRED Series | Type | Description |
| --- | --- | --- | --- |
| `vix` | `VIXCLS` | float | CBOE Volatility Index, used as a market uncertainty control. |
| `treasury_10y` | `DGS10` | float | 10-year Treasury constant maturity rate. |
| `treasury_2y` | `DGS2` | float | 2-year Treasury constant maturity rate. |
| `term_spread_10y_2y` | derived from `DGS10 - DGS2` | float | Yield curve slope between 10-year and 2-year Treasury rates. |
| `fed_funds_rate` | `FEDFUNDS` | float | Federal funds effective rate. |
| `cpi_all_items` | `CPIAUCSL` | float | Consumer Price Index for All Urban Consumers. |
| `cpi_yoy` | derived from `CPIAUCSL` | float | Year-over-year CPI growth rate. |
| `unemployment_rate` | `UNRATE` | float | U.S. unemployment rate. |
| `recession_indicator` | `USREC` | float | NBER recession indicator, where 1 indicates recession and 0 indicates non-recession. |

## Processed Files

| File | Description |
| --- | --- |
| `data/processed/headlines_clean.csv` | Raw headline rows after exact duplicate removal. |
| `data/processed/daily_dataset.csv` | Daily headline aggregation with next-day S&P 500 targets. |
| `data/processed/daily_with_macro.csv` | Main modeling dataset with macro controls joined. |
| `data/processed/eda_summary.json` | Machine-readable data quality and target summary. |
| `data/processed/eda_headlines_by_year.csv` | Year-level headline volume summary. |
| `data/processed/eda_regime_summary.csv` | Regime-level return, volatility, headline, and macro summary. |
| `data/processed/eda_macro_coverage.csv` | Macro variable coverage summary. |

## Known Limitations

- The original news collection process is not fully documented.
- The dataset uses daily closing prices only, so it cannot measure intraday price reactions.
- Headline volume increases sharply across the sample period, which can create concept drift.
- Duplicate headlines exist and should be removed or handled explicitly.
- Sentiment tools may misread financial context, sarcasm, or ambiguous market language.
- Monthly macro variables are joined using observation dates, not exact public release timestamps. This is acceptable for descriptive controls but should be handled carefully in predictive modeling.
