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

## Known Limitations

- The original news collection process is not fully documented.
- The dataset uses daily closing prices only, so it cannot measure intraday price reactions.
- Headline volume increases sharply across the sample period, which can create concept drift.
- Duplicate headlines exist and should be removed or handled explicitly.
- Sentiment tools may misread financial context, sarcasm, or ambiguous market language.
