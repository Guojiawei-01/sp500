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

## Prepared Daily Fields

| Field | Type | Description |
| --- | --- | --- |
| `Date` | date | Trading date after daily aggregation. |
| `CP` | float | S&P 500 closing price for the trading date. |
| `headline_count` | integer | Number of headlines on a trading date. |
| `unique_headline_count` | integer | Number of unique headlines on a trading date after duplicate checking. |
| `daily_text` | string | Headlines for the same date combined into one text field. |
| `close_next_day` | float | S&P 500 closing price on the next available trading date. |
| `return_next_day` | float | Next trading day's closing price divided by current closing price minus 1. |
| `direction_next_day` | integer | Binary target equal to 1 when `return_next_day` is positive and 0 otherwise. |
| `return_same_day` | float | Same-day close-to-close return from the previous available trading date to the current date. |
| `year` | integer | Calendar year extracted from `Date`. |
| `month` | string | Calendar month extracted from `Date` in `YYYY-MM` format. |
| `regime` | string | Market period label used for segmented analysis. |

## Macro Fields

Macro fields are downloaded from FRED, cached in `data/raw/macro/`, and joined to the daily headline table by most recent available observation date.

| Field | FRED Series | Type | Description |
| --- | --- | --- | --- |
| `vix` | `VIXCLS` | float | CBOE Volatility Index, used as market uncertainty context. |
| `treasury_10y` | `DGS10` | float | 10-year Treasury constant maturity rate. |
| `treasury_2y` | `DGS2` | float | 2-year Treasury constant maturity rate. |
| `term_spread_10y_2y` | derived from `DGS10 - DGS2` | float | Yield curve slope between 10-year and 2-year Treasury rates. |
| `fed_funds_rate` | `FEDFUNDS` | float | Federal funds effective rate. |
| `cpi_all_items` | `CPIAUCSL` | float | Consumer Price Index for All Urban Consumers. |
| `cpi_yoy` | derived from `CPIAUCSL` | float | Year-over-year CPI growth rate. |
| `unemployment_rate` | `UNRATE` | float | U.S. unemployment rate. |
| `recession_indicator` | `USREC` | float | NBER recession indicator, where 1 indicates recession and 0 indicates non-recession. |

## Sentiment Fields

The sentiment baseline is created in `notebooks/03_sentiment_analysis.ipynb` using transparent word and phrase dictionaries. These fields are intended as baseline features, not final NLP model outputs.

| Field | Type | Description |
| --- | --- | --- |
| `token_count` | integer | Number of parsed tokens in a headline. |
| `general_positive_count` | integer | Count of general positive dictionary terms in a headline. |
| `general_negative_count` | integer | Count of general negative dictionary terms in a headline. |
| `finance_positive_count` | integer | Count of finance-specific positive dictionary terms and phrases in a headline. |
| `finance_negative_count` | integer | Count of finance-specific negative dictionary terms and phrases in a headline. |
| `general_sentiment_score` | float | Headline-level general score, computed as positive minus negative counts divided by token count. |
| `finance_sentiment_score` | float | Headline-level finance score, computed as positive minus negative finance counts divided by token count. |
| `finance_sentiment_label` | string | Headline label: `positive`, `negative`, or `neutral`, based on the finance sentiment score. |
| `scored_headline_count` | integer | Number of scored headlines on a trading date. |
| `total_token_count` | integer | Total parsed headline tokens on a trading date. |
| `avg_general_sentiment` | float | Average headline-level general sentiment score on a trading date. |
| `avg_finance_sentiment` | float | Average headline-level finance sentiment score on a trading date. |
| `finance_sentiment_std` | float | Standard deviation of headline-level finance sentiment scores on a trading date. |
| `positive_headline_share` | float | Share of headlines labeled positive on a trading date. |
| `negative_headline_share` | float | Share of headlines labeled negative on a trading date. |
| `neutral_headline_share` | float | Share of headlines labeled neutral on a trading date. |
| `net_positive_negative_share` | float | Positive headline share minus negative headline share on a trading date. |

## Processed Files

| File | Description |
| --- | --- |
| `data/processed/headlines_clean.csv` | Raw headline rows after exact duplicate removal. |
| `data/processed/daily_dataset.csv` | Daily headline aggregation with next-day S&P 500 targets. |
| `data/processed/daily_with_macro.csv` | Prepared daily dataset with macro context fields joined for EDA. |
| `data/processed/headlines_with_sentiment.csv` | Headline-level dictionary sentiment scores. |
| `data/processed/daily_with_sentiment.csv` | Daily market, macro, target, and sentiment features for modeling. |
| `data/processed/sentiment_summary.json` | Machine-readable sentiment analysis summary. |
| `data/processed/sentiment_correlation_summary.csv` | Descriptive correlations between sentiment features and next-day outcomes. |
| `data/processed/sentiment_tercile_summary.csv` | Next-day outcomes grouped by daily finance sentiment tercile. |
| `data/processed/sentiment_regime_summary.csv` | Sentiment and return summary by market regime. |
| `data/processed/eda_summary.json` | Machine-readable data quality and target summary. |
| `data/processed/eda_headlines_by_year.csv` | Year-level headline volume summary. |
| `data/processed/eda_duplicates_by_year.csv` | Year-level count of exact duplicate rows removed. |
| `data/processed/eda_return_extremes.csv` | Largest absolute next-day return dates for manual EDA review. |
| `data/processed/eda_possible_off_topic_sample.csv` | Simple keyword-based sample of headlines for manual quality review. |
| `data/processed/eda_regime_summary.csv` | Regime-level return, volatility, headline, and macro summary. |
| `data/processed/eda_macro_coverage.csv` | Macro variable coverage summary. |

## Known Limitations

- The original news collection process is not fully documented.
- The dataset uses daily closing prices only, so it cannot measure intraday price reactions.
- Headline volume increases sharply across the sample period, which can create concept drift.
- Duplicate headlines exist and should be removed or handled explicitly.
- Monthly macro variables are joined using observation dates, not exact public release timestamps. They are included here for descriptive EDA context.
- Dictionary sentiment is transparent and reproducible, but it can misread context, sarcasm, and finance-specific phrasing. It should be treated as a baseline.
