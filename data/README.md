# Data

This folder stores raw and processed project data.

```text
data/raw/sp500_headlines_2008_2024.csv
data/raw/macro/
data/processed/
```

The dataset comes from Kaggle: **S&P 500 with Financial News Headlines (2008-2024)** by dyutidasmahaptra, licensed under CC BY-SA 4.0.

Current file summary:

- Rows: 19,127
- Columns: 3
- Date range: 2008-01-02 to 2024-03-04
- Unique trading dates: 3,507
- Missing values: none in the current file
- Duplicate full rows: 974

The raw headline file should not be edited directly. FRED macro files are downloaded into `data/raw/macro/` by:

```bash
python scripts/prepare_data.py
```

Generated processed files:

- `headlines_clean.csv`: raw headline rows after exact duplicate removal
- `daily_dataset.csv`: daily headline aggregation with next-day return targets
- `daily_with_macro.csv`: prepared daily table with FRED macro controls for EDA
- `eda_summary.json`: machine-readable data preparation summary
- `eda_headlines_by_year.csv`: yearly headline count summary
- `eda_regime_summary.csv`: regime-level EDA summary
- `eda_macro_coverage.csv`: macro variable coverage rates
