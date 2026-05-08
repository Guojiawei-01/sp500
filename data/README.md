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

The raw headline file should not be edited directly. FRED macro files are cached in `data/raw/macro/`. The preparation script uses those cached files by default:

```bash
python scripts/prepare_data.py
```

To download fresh FRED files, run:

```bash
python scripts/prepare_data.py --refresh-macro
```

Generated processed files:

- `headlines_clean.csv`: raw headline rows after exact duplicate removal
- `daily_dataset.csv`: daily headline aggregation with next-day return targets
- `daily_with_macro.csv`: prepared daily table with FRED macro context fields for EDA
- `headlines_with_sentiment.csv`: headline-level dictionary sentiment scores
- `daily_with_sentiment.csv`: daily table with macro context, targets, and sentiment features for modeling
- `sentiment_summary.json`: machine-readable sentiment analysis summary
- `sentiment_correlation_summary.csv`: descriptive correlations between sentiment features and next-day outcomes
- `sentiment_tercile_summary.csv`: next-day outcomes grouped by daily finance sentiment tercile
- `sentiment_regime_summary.csv`: sentiment and return summary by market regime
- `eda_summary.json`: machine-readable data preparation summary
- `eda_headlines_by_year.csv`: yearly headline count summary
- `eda_duplicates_by_year.csv`: duplicate rows removed by year
- `eda_return_extremes.csv`: largest absolute next-day return dates
- `eda_possible_off_topic_sample.csv`: simple keyword-based sample for manual headline quality review
- `eda_regime_summary.csv`: regime-level EDA summary
- `eda_macro_coverage.csv`: macro variable coverage rates
- `headlines_with_sentiment_v2.csv`: headline-level VADER, FinBERT, and LDA topic features
- `daily_with_sentiment_v2.csv`: daily table with dictionary, VADER, FinBERT, topic, macro, and target fields
- `model_predictions.csv`: calibrated Logit/XGB predictions with validation-selected thresholds
- `model_metrics_summary.csv`: model metrics including balanced accuracy, AUC, F1, and Brier score
- `model_thresholds.csv`: thresholds selected on validation balanced accuracy
- `model_tuning_summary.csv`: XGBoost validation tuning grid
- `model_regime_metrics.csv`: classification metrics by regime
- `backtest_performance_summary.csv`: long/flat strategy summary by split/model/feature set
- `backtest_regime_summary.csv`: backtest summary by market regime
- `robustness_threshold_cost_summary.csv`: threshold and transaction-cost robustness checks
- `model_failure_cases.csv`: selected failure cases for report discussion
- `return_model_metrics_summary.csv`: continuous next-day return model metrics
- `return_model_predictions.csv`: test-set return predictions used for figure 9
