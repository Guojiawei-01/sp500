# Does Financial News Sentiment Move the S&P 500?

DATA 498D capstone project exploring whether daily financial news headlines contain useful information about next-day S&P 500 movements.

## Project Question

Financial markets are often assumed to incorporate public information quickly. This project tests whether news headline sentiment is associated with short-run S&P 500 direction and whether that relationship changes across different market conditions.

Main questions:

1. Is headline sentiment statistically related to next-day S&P 500 returns?
2. Do finance-specific sentiment methods perform better than general sentiment tools?
3. Does predictive performance differ across calm and high-volatility market regimes?

## Data

The primary dataset is the Kaggle dataset **S&P 500 with Financial News Headlines (2008-2024)** by dyutidasmahaptra, licensed under CC BY-SA 4.0.

Included raw file:

```text
data/raw/sp500_headlines_2008_2024.csv
```

The working CSV contains 19,127 rows, 3 columns, and 3,507 unique trading dates from January 2, 2008 through March 4, 2024.

Important data limitations:

- The original news source is not fully documented.
- Some headlines may be off-topic or only loosely related to the S&P 500.
- The dataset has daily closing prices only, so intraday reactions cannot be measured.
- Headline volume changes strongly over time, especially after 2019.
- The current file includes duplicate rows that should be handled before modeling.

## Repository Structure

```text
.
├── README.md
├── CONTRIBUTIONS.md
├── data_dictionary.md
├── requirements.txt
├── new.pdf
├── data/
│   ├── raw/
│   │   └── sp500_headlines_2008_2024.csv
│   └── processed/
├── figures/
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_sentiment_analysis.ipynb
│   ├── 04_modeling.ipynb
│   └── 05_backtest_regime_analysis.ipynb
└── report/
```

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Open and run the notebooks in order:

```text
notebooks/01_data_cleaning.ipynb
notebooks/02_eda.ipynb
notebooks/03_sentiment_analysis.ipynb
notebooks/04_modeling.ipynb
notebooks/05_backtest_regime_analysis.ipynb
```

The notebooks load the CSV from `data/raw/sp500_headlines_2008_2024.csv`.

## Analysis Plan

The analysis is organized around these steps:

- `01_data_cleaning.ipynb`: validate raw data, remove duplicates, create daily targets
- `02_eda.ipynb`: explore headline volume, price movement, returns, and data quality
- `03_sentiment_analysis.ipynb`: create baseline sentiment features
- `04_modeling.ipynb`: test predictive models for next-day direction and return
- `05_backtest_regime_analysis.ipynb`: evaluate regimes and simple trading signal performance

## Deliverables Checklist

- [x] Project proposal
- [x] Dataset added to repository
- [x] Data dictionary
- [x] Reproducible analysis notebook scaffold
- [x] README with run instructions
- [ ] Final analysis results
- [ ] Written report
- [ ] Recorded presentation
- [ ] Peer review submission

## Note on Written Work

The final report and individual reflection paragraphs should be written by team members in their own words, following the course academic integrity policy.
