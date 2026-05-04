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

Macro controls are downloaded from FRED by `scripts/prepare_data.py`:

- `VIXCLS`: CBOE Volatility Index
- `DGS10`: 10-year Treasury constant maturity rate
- `DGS2`: 2-year Treasury constant maturity rate
- `FEDFUNDS`: Federal funds effective rate
- `CPIAUCSL`: Consumer Price Index for All Urban Consumers
- `UNRATE`: Unemployment rate
- `USREC`: NBER recession indicator

Important data limitations:

- The original news source is not fully documented.
- Some headlines may be off-topic or only loosely related to the S&P 500.
- The dataset has daily closing prices only, so intraday reactions cannot be measured.
- Headline volume changes strongly over time, especially after 2019.
- The current file includes duplicate rows that are removed during cleaning.
- Monthly macro variables are joined by most recent available observation date for EDA context.

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
├── scripts/
│   └── prepare_data.py
└── report/
```

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the data preparation pipeline:

```bash
python scripts/prepare_data.py
```

For the current data cleaning and EDA step, run:

```text
notebooks/01_data_cleaning.ipynb
notebooks/02_eda.ipynb
```

The first notebook also runs the same preparation script. The prepared daily table is:

```text
data/processed/daily_with_macro.csv
```

## Analysis Plan

Current completed step:

- `01_data_cleaning.ipynb`: validate raw data, remove duplicates, create daily targets, join macro controls
- `02_eda.ipynb`: explore headline volume, price movement, returns, macro controls, and regimes

The remaining notebooks are placeholders for later team members.

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
