# Backtest, Regime Evaluation, and Return Modeling Summary

This report completes the proposal's evaluation-lead workstream and adds the continuous next-day return modeling handoff.

## Inputs

- `daily_with_sentiment_v2.csv`: daily market, macro, dictionary, VADER, FinBERT, and LDA topic features.
- `model_predictions.csv`: validation, test, and expanding-window out-of-sample direction predictions.
- Default backtest cost: 0.0001 per signal change (1 basis point).

## Regime Classification Performance

Expanding-window all-feature direction models by regime:

| model_name   | regime                     |    n |   accuracy |    auc |     f1 |   up_day_rate |
|:-------------|:---------------------------|-----:|-----------:|-------:|-------:|--------------:|
| Logit        | 2008-2009 financial crisis |   90 |     0.4    | 0.3412 | 0      |        0.5889 |
| Logit        | 2020 COVID shock           |  249 |     0.5542 | 0.4766 | 0.7024 |        0.5663 |
| Logit        | 2022-2023 rate-hike cycle  |  495 |     0.4909 | 0.4446 | 0.5655 |        0.4869 |
| Logit        | Other                      | 2566 |     0.5016 | 0.4815 | 0.5833 |        0.5518 |
| XGB          | 2008-2009 financial crisis |   90 |     0.4333 | 0.386  | 0.4    |        0.5889 |
| XGB          | 2020 COVID shock           |  249 |     0.5261 | 0.5018 | 0.6685 |        0.5663 |
| XGB          | 2022-2023 rate-hike cycle  |  495 |     0.4727 | 0.459  | 0.4314 |        0.4869 |
| XGB          | Other                      | 2566 |     0.5144 | 0.5127 | 0.5753 |        0.5518 |

## Long/Flat Backtest

The strategy is long the S&P 500 next day when predicted up-day probability is at least 0.50 and flat otherwise. Net returns subtract transaction costs when the signal changes.

| model_name   |   strategy_cumulative_return |   buy_hold_cumulative_return |   strategy_sharpe |   strategy_max_drawdown |   exposure |   trades |
|:-------------|-----------------------------:|-----------------------------:|------------------:|------------------------:|-----------:|---------:|
| Logit        |                       0.9475 |                       4.6592 |            0.4005 |                 -0.2993 |     0.6544 |      915 |
| XGB          |                       1.3373 |                       4.6592 |            0.4808 |                 -0.3583 |     0.5832 |     1347 |

## Robustness

Threshold and transaction-cost robustness is saved to `data/processed/robustness_threshold_cost_summary.csv` and visualized in `figures/fig8_backtest_robustness.png`.

Best robustness rows by cumulative return:

| model_name   |   threshold |   transaction_cost |   cumulative_return |   sharpe |   max_drawdown |   exposure |   trades |
|:-------------|------------:|-------------------:|--------------------:|---------:|---------------:|-----------:|---------:|
| Logit        |        0.45 |             0      |              2.3085 |   0.6182 |        -0.3392 |     0.8182 |      573 |
| XGB          |        0.45 |             0      |              2.1929 |   0.6048 |        -0.3394 |     0.685  |     1159 |
| Logit        |        0.45 |             0.0001 |              2.1243 |   0.5926 |        -0.3392 |     0.8182 |      573 |
| XGB          |        0.45 |             0.0001 |              1.8436 |   0.5528 |        -0.34   |     0.685  |     1159 |
| XGB          |        0.5  |             0      |              1.6742 |   0.5446 |        -0.3487 |     0.5832 |     1347 |
| XGB          |        0.55 |             0      |              1.6706 |   0.5709 |        -0.3614 |     0.4759 |     1371 |

## Continuous Return Modeling

The return-modeling supplement predicts continuous `return_next_day` with Ridge and XGBRegressor under the same time-aware split used by the classification models.

| model        | feature_set   |    rmse |     mae |       r2 |   information_coefficient |   direction_accuracy |
|:-------------|:--------------|--------:|--------:|---------:|--------------------------:|---------------------:|
| Ridge        | all           | 0.01207 | 0.00913 | -0.01782 |                   0.08246 |              0.52328 |
| Ridge        | general       | 0.01198 | 0.00902 | -0.00351 |                   0.07837 |              0.49534 |
| Ridge        | vader         | 0.012   | 0.00903 | -0.00555 |                   0.04301 |              0.49348 |
| Ridge        | dict          | 0.01202 | 0.00906 | -0.00995 |                   0.03855 |              0.49348 |
| XGBRegressor | all           | 0.01213 | 0.00906 | -0.02806 |                   0.01269 |              0.50279 |

## Failure Cases

Failure cases include false long losses, missed rallies, and high-confidence classification errors. They are saved to `data/processed/model_failure_cases.csv` for manual discussion in the final report.

| failure_type         | Date                | regime                     | model_name   |   y_pred_proba |   return_next_day |   headline_count |   vix |
|:---------------------|:--------------------|:---------------------------|:-------------|---------------:|------------------:|-----------------:|------:|
| classification_error | 2009-02-23 00:00:00 | 2008-2009 financial crisis | Logit        |         0.0014 |            0.029  |                1 | 52.62 |
| classification_error | 2009-03-03 00:00:00 | 2008-2009 financial crisis | Logit        |         0.0011 |            0.0238 |                1 | 50.93 |
| classification_error | 2009-04-06 00:00:00 | 2008-2009 financial crisis | Logit        |         0.0032 |            0.0278 |                2 | 40.93 |
| classification_error | 2009-05-07 00:00:00 | 2008-2009 financial crisis | Logit        |         0.0037 |            0.002  |                1 | 33.44 |
| classification_error | 2009-09-02 00:00:00 | 2008-2009 financial crisis | Logit        |         0.0038 |            0.0308 |                1 | 28.9  |
| classification_error | 2009-09-14 00:00:00 | 2008-2009 financial crisis | Logit        |         0.0005 |            0.0185 |                1 | 23.86 |
| classification_error | 2009-11-12 00:00:00 | 2008-2009 financial crisis | Logit        |         0.0008 |            0.0057 |                1 | 24.24 |
| classification_error | 2009-11-13 00:00:00 | 2008-2009 financial crisis | Logit        |         0.0004 |            0.0117 |                3 | 23.36 |

## Interpretation

- Direction models have weak and unstable predictive power by regime, especially in crisis periods.
- Long/flat economic value is sensitive to threshold choice and transaction costs.
- Continuous return models add the missing proposal component, but their out-of-sample R-squared remains weak; use them as robustness evidence rather than as a trading claim.