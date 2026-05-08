# Contributions

This file summarizes the main responsibilities each team member owned during the project. The work was collaborative, so several decisions were reviewed by more than one person before being finalized.

## Team Roles

| Team Member | Main Responsibilities | Notes |
| --- | --- | --- |
| Jiawei Guo | Data preparation lead: raw data validation, duplicate removal, daily aggregation, next-day target construction, FRED macro integration, data dictionary, and EDA summaries | Also documented key data limitations such as headline-density drift, duplicate handling, and daily close-price timing |
| Chaoran Chen | NLP lead: dictionary sentiment baseline, VADER and FinBERT sentiment scoring, LDA topic features, and comparison of sentiment feature sets | Helped interpret why stronger language models did not create a large daily prediction edge |
| Haoyang Liu | Modeling lead: feature engineering, time-aware train/validation/test split, Logistic Regression and XGBoost modeling, validation-selected thresholds, calibration checks, and return-model supplement | Focused on direction prediction metrics, baseline comparison, and avoiding leakage in the modeling pipeline |
| Haonan Li | Evaluation lead: long/flat backtest design, regime-level evaluation, transaction-cost and threshold robustness, exposure checks, and failure-case review | Helped separate visually promising backtest results from signals that survived stricter evaluation |
| Duli Lei | Writing and presentation lead: report narrative, figure organization, final takeaways, presentation flow, and interactive dashboard preparation | Helped turn the analysis into a clear story for the written report and recorded presentation |

## AI Assistance Disclosure

AI tools were used as support for repository organization, debugging, documentation drafting, and presentation/dashboard polishing. Final project decisions, interpretation of results, individual reflections, and submitted materials should be reviewed and approved by the team.
