# Data

This folder stores raw and processed project data.

```text
data/raw/sp500_headlines_2008_2024.csv
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

The raw file should not be edited directly. Cleaned or feature-engineered files can be saved in `data/processed/`.
