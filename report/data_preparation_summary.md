# Data Preparation Summary

## Raw Data Quality

- Raw rows: 19,127
- Clean rows after exact duplicate removal: 18,153
- Exact duplicate rows removed: 974
- Date range: 2008-01-02 to 2024-03-04
- Unique trading dates in raw data: 3,507
- Dates with multiple S&P 500 close values: 0
- Missing values in raw data: {'Title': 0, 'Date': 0, 'CP': 0}

## Daily Modeling Table

- Unique trading dates after target creation: 3,506
- Average headlines per trading day: 5.17
- Median headlines per trading day: 4
- Maximum headlines on one trading day: 55
- Mean next-day return: 0.0453%
- Next-day return standard deviation: 1.3498%
- Up-day rate: 54.3%

## Macro Coverage

- `vix`: 100.0% coverage
- `treasury_10y`: 100.0% coverage
- `treasury_2y`: 100.0% coverage
- `term_spread_10y_2y`: 100.0% coverage
- `fed_funds_rate`: 100.0% coverage
- `cpi_all_items`: 100.0% coverage
- `cpi_yoy`: 100.0% coverage
- `unemployment_rate`: 100.0% coverage
- `recession_indicator`: 100.0% coverage

## Notes for Modeling

- Exact duplicate rows are removed before daily aggregation.
- The target is next trading day's return, not same-day return.
- FRED macro variables are joined by most recent available observation date using an as-of join.
- Monthly macro variables may not perfectly reflect publication timing; treat them as controls unless release lags are modeled explicitly.
