"""Evaluate model predictions, backtests, robustness, and return models."""

from __future__ import annotations

import sys
import warnings
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = PROJECT_ROOT / ".cache"
CACHE_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from modeling_pipeline import load_and_engineer_features, safe_auc, split_by_date  # noqa: E402


DATA = PROJECT_ROOT / "data" / "processed"
FIG = PROJECT_ROOT / "figures"
REPORT = PROJECT_ROOT / "report"
FIG.mkdir(exist_ok=True)
REPORT.mkdir(exist_ok=True)

DEFAULT_COST = 0.0001
ROBUSTNESS_THRESHOLDS = [0.40, 0.45, 0.50, 0.55, 0.60]
ROBUSTNESS_COSTS = [0.0, 0.0001, 0.0005]
RANDOM_STATE = 42


def safe_group_auc(y_true: pd.Series, y_proba: pd.Series) -> float:
    if y_true.nunique() < 2:
        return np.nan
    return float(roc_auc_score(y_true, y_proba))


def classification_metrics(group: pd.DataFrame) -> dict[str, float]:
    return {
        "n": len(group),
        "up_day_rate": group["y_true"].mean(),
        "accuracy": accuracy_score(group["y_true"], group["y_pred"]),
        "balanced_accuracy": balanced_accuracy_score(group["y_true"], group["y_pred"]),
        "precision": precision_score(group["y_true"], group["y_pred"], zero_division=0),
        "recall": recall_score(group["y_true"], group["y_pred"], zero_division=0),
        "f1": f1_score(group["y_true"], group["y_pred"], zero_division=0),
        "auc": safe_group_auc(group["y_true"], group["y_pred_proba"]),
        "brier": brier_score_loss(group["y_true"], group["y_pred_proba"]),
        "avg_next_return": group["return_next_day"].mean(),
        "volatility": group["return_next_day"].std(),
        "threshold": group["threshold"].median() if "threshold" in group else 0.5,
    }


def max_drawdown(returns: pd.Series) -> float:
    if returns.empty:
        return np.nan
    equity = (1 + returns).cumprod()
    running_max = equity.cummax()
    return float((equity / running_max - 1).min())


def performance_summary(returns: pd.Series) -> dict[str, float]:
    returns = returns.fillna(0)
    n = len(returns)
    cumulative = float((1 + returns).prod() - 1) if n else np.nan
    annualized_return = float((1 + cumulative) ** (252 / n) - 1) if n and cumulative > -1 else np.nan
    annualized_vol = float(returns.std() * np.sqrt(252)) if n else np.nan
    sharpe = float((returns.mean() / returns.std()) * np.sqrt(252)) if returns.std() and returns.std() > 0 else np.nan
    return {
        "n_days": n,
        "cumulative_return": cumulative,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown(returns),
        "avg_daily_return": float(returns.mean()) if n else np.nan,
    }


def apply_backtest(group: pd.DataFrame, threshold: float | None = None, transaction_cost: float = DEFAULT_COST) -> pd.DataFrame:
    out = group.sort_values("Date").copy()
    threshold_series = out["threshold"] if threshold is None and "threshold" in out else threshold
    if isinstance(threshold_series, pd.Series):
        out["signal"] = (out["y_pred_proba"] >= threshold_series.astype(float)).astype(int)
        out["threshold_used"] = threshold_series.astype(float)
    else:
        threshold_value = 0.5 if threshold_series is None else float(threshold_series)
        out["signal"] = (out["y_pred_proba"] >= threshold_value).astype(int)
        out["threshold_used"] = threshold_value

    out["trade"] = out["signal"].diff().abs().fillna(out["signal"]).astype(float)
    out["strategy_return_net"] = out["signal"] * out["return_next_day"] - transaction_cost * out["trade"]
    out["buy_hold_return"] = out["return_next_day"]
    return out


def summarize_backtest(group: pd.DataFrame, threshold: float | None = None, transaction_cost: float = DEFAULT_COST) -> dict:
    bt = apply_backtest(group, threshold=threshold, transaction_cost=transaction_cost)
    strategy = performance_summary(bt["strategy_return_net"])
    buy_hold = performance_summary(bt["buy_hold_return"])
    out = {
        "threshold": float(bt["threshold_used"].median()),
        "transaction_cost": transaction_cost,
        "strategy_n_days": strategy["n_days"],
        "strategy_cumulative_return": strategy["cumulative_return"],
        "strategy_annualized_return": strategy["annualized_return"],
        "strategy_annualized_volatility": strategy["annualized_volatility"],
        "strategy_sharpe": strategy["sharpe"],
        "strategy_max_drawdown": strategy["max_drawdown"],
        "strategy_avg_daily_return": strategy["avg_daily_return"],
        "buy_hold_n_days": buy_hold["n_days"],
        "buy_hold_cumulative_return": buy_hold["cumulative_return"],
        "buy_hold_annualized_return": buy_hold["annualized_return"],
        "buy_hold_annualized_volatility": buy_hold["annualized_volatility"],
        "buy_hold_sharpe": buy_hold["sharpe"],
        "buy_hold_max_drawdown": buy_hold["max_drawdown"],
        "buy_hold_avg_daily_return": buy_hold["avg_daily_return"],
        "exposure": float(bt["signal"].mean()),
        "trades": int(bt["trade"].sum()),
    }
    if bt["signal"].sum() > 0:
        out["win_rate_when_in_market"] = float((bt.loc[bt["signal"] == 1, "return_next_day"] > 0).mean())
    else:
        out["win_rate_when_in_market"] = np.nan
    return out


def build_regime_metrics(preds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["split", "model_name", "feature_set", "regime"]
    for keys, group in preds.groupby(group_cols):
        row = dict(zip(group_cols, keys))
        row.update(classification_metrics(group))
        rows.append(row)
    return pd.DataFrame(rows)


def build_backtests(preds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    regime_rows = []
    group_cols = ["split", "model_name", "feature_set"]
    for keys, group in preds.groupby(group_cols):
        row = dict(zip(group_cols, keys))
        row.update(summarize_backtest(group))
        summary_rows.append(row)

        for regime, regime_group in group.groupby("regime"):
            rrow = dict(zip(group_cols, keys))
            rrow["regime"] = regime
            rrow.update(summarize_backtest(regime_group))
            regime_rows.append(rrow)

    robustness_rows = []
    base = preds[
        (preds["feature_set"] == "all")
        & (preds["model_name"].isin(["Logit", "XGB"]))
        & (preds["split"].isin(["val", "test", "oos_expanding"]))
    ].copy()
    for (split, model_name), group in base.groupby(["split", "model_name"]):
        for threshold in ROBUSTNESS_THRESHOLDS:
            for cost in ROBUSTNESS_COSTS:
                row = {
                    "split": split,
                    "model_name": model_name,
                    "feature_set": "all",
                }
                row.update(summarize_backtest(group, threshold=threshold, transaction_cost=cost))
                robustness_rows.append(row)

    return pd.DataFrame(summary_rows), pd.DataFrame(regime_rows), pd.DataFrame(robustness_rows)


def equity_curve(group: pd.DataFrame, label: str) -> pd.DataFrame:
    bt = apply_backtest(group)
    return pd.DataFrame(
        {
            "Date": bt["Date"],
            "equity": (1 + bt["strategy_return_net"]).cumprod(),
            "buy_hold": (1 + bt["buy_hold_return"]).cumprod(),
            "model": label,
        }
    )


def plot_backtest_equity(preds: pd.DataFrame) -> None:
    base = preds[
        (preds["split"] == "test")
        & (preds["feature_set"] == "all")
        & (preds["model_name"].isin(["Logit", "XGB"]))
    ]
    if base.empty:
        base = preds[
            (preds["split"] == "oos_expanding")
            & (preds["feature_set"] == "all")
            & (preds["model_name"].isin(["Logit", "XGB"]))
        ]

    curves = [equity_curve(group, name) for name, group in base.groupby("model_name")]
    curve_df = pd.concat(curves, ignore_index=True)
    buy_hold = curve_df[["Date", "buy_hold"]].drop_duplicates()

    fig, ax = plt.subplots(figsize=(10, 5))
    for model_name, group in curve_df.groupby("model"):
        ax.plot(group["Date"], group["equity"], label=f"{model_name} strategy")
    ax.plot(buy_hold["Date"], buy_hold["buy_hold"], color="black", ls="--", label="Buy and hold")
    ax.set_title("Long/Flat Equity Curves - Test Set Thresholds from Validation")
    ax.set_ylabel("Growth of $1")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG / "fig6_backtest_equity_curves.png", dpi=140)
    plt.close(fig)


def plot_regime_returns(backtest_regime: pd.DataFrame) -> None:
    base = backtest_regime[
        (backtest_regime["split"] == "oos_expanding")
        & (backtest_regime["feature_set"] == "all")
        & (backtest_regime["model_name"].isin(["Logit", "XGB"]))
    ].copy()
    if base.empty:
        return

    regimes = list(base["regime"].drop_duplicates())
    x = np.arange(len(regimes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(11, 5))
    for i, model_name in enumerate(["Logit", "XGB"]):
        vals = [
            base[(base["model_name"] == model_name) & (base["regime"] == regime)][
                "strategy_cumulative_return"
            ].iloc[0]
            for regime in regimes
        ]
        ax.bar(x + (i - 0.5) * width, vals, width, label=f"{model_name} strategy")
    buy_vals = [
        base[base["regime"] == regime]["buy_hold_cumulative_return"].iloc[0]
        for regime in regimes
    ]
    ax.scatter(x + width, buy_vals, marker="D", color="black", label="Buy and hold")
    ax.axhline(0, color="black", lw=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(regimes, rotation=20, ha="right")
    ax.set_title("OOS Backtest Returns by Regime")
    ax.set_ylabel("Cumulative return")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG / "fig7_regime_backtest_returns.png", dpi=140)
    plt.close(fig)


def plot_robustness(robustness: pd.DataFrame) -> None:
    base = robustness[
        (robustness["split"] == "val")
        & (robustness["feature_set"] == "all")
        & (robustness["transaction_cost"].isin([0.0, DEFAULT_COST, 0.0005]))
    ].copy()
    if base.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, model_name in zip(axes, ["Logit", "XGB"]):
        sub = base[base["model_name"] == model_name]
        for cost, group in sub.groupby("transaction_cost"):
            group = group.sort_values("threshold")
            ax.plot(group["threshold"], group["strategy_cumulative_return"], marker="o", label=f"cost={cost:g}")
        ax.axhline(0, color="black", lw=0.7)
        ax.set_title(model_name)
        ax.set_xlabel("Validation threshold")
        ax.set_ylabel("Cumulative return")
        ax.legend(fontsize=8)
    fig.suptitle("Validation Threshold and Cost Robustness")
    plt.tight_layout()
    plt.savefig(FIG / "fig8_backtest_robustness.png", dpi=140)
    plt.close(fig)


def build_failure_cases(preds: pd.DataFrame) -> pd.DataFrame:
    daily_path = DATA / "daily_with_sentiment_v2.csv"
    daily = pd.read_csv(daily_path, parse_dates=["Date"])[["Date", "headline_count", "vix", "daily_text"]]
    base = preds[
        (preds["split"] == "test")
        & (preds["feature_set"] == "all")
        & (preds["model_name"].isin(["Logit", "XGB"]))
    ].merge(daily, on="Date", how="left")

    pieces = []
    errors = base[base["y_pred"] != base["y_true"]].copy()
    errors["failure_type"] = "classification_error"
    errors["confidence"] = (errors["y_pred_proba"] - errors["threshold"]).abs()
    pieces.append(errors.sort_values("confidence", ascending=False).head(12))

    false_long = base[(base["y_pred"] == 1) & (base["return_next_day"] < 0)].copy()
    false_long["failure_type"] = "false_long_loss"
    pieces.append(false_long.sort_values("return_next_day").head(12))

    missed = base[(base["y_pred"] == 0) & (base["return_next_day"] > 0)].copy()
    missed["failure_type"] = "missed_rally"
    pieces.append(missed.sort_values("return_next_day", ascending=False).head(12))

    out = pd.concat(pieces, ignore_index=True)
    out["signal"] = out["y_pred"]
    out["strategy_return_net"] = out["signal"] * out["return_next_day"]
    out["daily_text_snippet"] = out["daily_text"].astype(str).str.slice(0, 280)
    keep = [
        "failure_type",
        "Date",
        "regime",
        "model_name",
        "feature_set",
        "y_true",
        "y_pred",
        "y_pred_proba",
        "threshold",
        "signal",
        "return_next_day",
        "strategy_return_net",
        "headline_count",
        "vix",
        "daily_text_snippet",
    ]
    return out[keep].sort_values(["failure_type", "Date", "model_name"]).reset_index(drop=True)


def information_coefficient(y_true: np.ndarray, pred: np.ndarray) -> float:
    if np.std(pred) == 0 or np.std(y_true) == 0:
        return np.nan
    return float(np.corrcoef(y_true, pred)[0, 1])


def return_metric_row(model: str, feature_set: str, split: str, y_true: np.ndarray, pred: np.ndarray) -> dict:
    rmse = float(np.sqrt(np.mean((y_true - pred) ** 2)))
    return {
        "model": model,
        "feature_set": feature_set,
        "split": split,
        "n": len(y_true),
        "rmse": rmse,
        "mae": mean_absolute_error(y_true, pred),
        "r2": r2_score(y_true, pred),
        "information_coefficient": information_coefficient(y_true, pred),
        "direction_accuracy": accuracy_score(y_true > 0, pred > 0),
        "predicted_long_share": float((pred > 0).mean()),
        "actual_up_day_rate": float((y_true > 0).mean()),
    }


def build_return_models() -> tuple[pd.DataFrame, pd.DataFrame]:
    df_model, feature_sets, _ = load_and_engineer_features()
    train_df, val_df, test_df = split_by_date(df_model)
    rows = []
    prediction_parts = []

    for split_name, split_df in [("val", val_df), ("test", test_df)]:
        y_true = split_df["return_next_day"].values
        pred = np.full(len(split_df), train_df["return_next_day"].mean())
        rows.append(return_metric_row("HistoricalMean", "-", split_name, y_true, pred))

    for feature_set, feats in feature_sets.items():
        scaler = StandardScaler().fit(train_df[feats])
        x_train = scaler.transform(train_df[feats])
        y_train = train_df["return_next_day"].values

        ridge = Ridge(alpha=1.0)
        ridge.fit(x_train, y_train)
        xgb = XGBRegressor(
            n_estimators=200,
            max_depth=2,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
        )
        xgb.fit(train_df[feats], y_train)

        for split_name, split_df in [("val", val_df), ("test", test_df)]:
            y_true = split_df["return_next_day"].values
            ridge_pred = ridge.predict(scaler.transform(split_df[feats]))
            xgb_pred = xgb.predict(split_df[feats])
            rows.append(return_metric_row("Ridge", feature_set, split_name, y_true, ridge_pred))
            rows.append(return_metric_row("XGBRegressor", feature_set, split_name, y_true, xgb_pred))

            if split_name == "test" and feature_set == "all":
                prediction_parts.append(
                    pd.DataFrame(
                        {
                            "Date": split_df["Date"].values,
                            "return_next_day": y_true,
                            "ridge_pred": ridge_pred,
                            "xgb_pred": xgb_pred,
                        }
                    )
                )

    metrics = pd.DataFrame(rows)
    predictions = pd.concat(prediction_parts, ignore_index=True)
    return metrics, predictions


def plot_return_predictions(return_predictions: pd.DataFrame) -> None:
    if return_predictions.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(return_predictions["Date"], return_predictions["return_next_day"], label="Actual", lw=0.8)
    axes[0].plot(return_predictions["Date"], return_predictions["ridge_pred"], label="Ridge all", lw=0.8)
    axes[0].axhline(0, color="black", lw=0.6)
    axes[0].set_title("Test Return Prediction Over Time")
    axes[0].legend(fontsize=8)

    axes[1].scatter(return_predictions["return_next_day"], return_predictions["ridge_pred"], s=14, alpha=0.6)
    axes[1].axhline(0, color="black", lw=0.6)
    axes[1].axvline(0, color="black", lw=0.6)
    axes[1].set_title("Actual vs Predicted Returns")
    axes[1].set_xlabel("Actual next-day return")
    axes[1].set_ylabel("Predicted next-day return")
    plt.tight_layout()
    plt.savefig(FIG / "fig9_return_model_predictions.png", dpi=140)
    plt.close(fig)


def write_summary(
    regime_metrics: pd.DataFrame,
    backtest_summary: pd.DataFrame,
    robustness: pd.DataFrame,
    return_metrics: pd.DataFrame,
) -> None:
    test_metrics = regime_metrics[
        (regime_metrics["split"] == "test")
        & (regime_metrics["feature_set"] == "all")
        & (regime_metrics["model_name"].isin(["Logit", "XGB"]))
    ]
    test_backtest = backtest_summary[
        (backtest_summary["split"] == "test")
        & (backtest_summary["feature_set"] == "all")
        & (backtest_summary["model_name"].isin(["Logit", "XGB"]))
    ]
    best_robust = robustness.sort_values("strategy_cumulative_return", ascending=False).head(8)
    test_return = return_metrics[return_metrics["split"] == "test"].sort_values(
        "information_coefficient", ascending=False
    ).head(8)

    text = f"""# Backtest, Regime Evaluation, and Return Modeling Summary

This report uses calibrated direction-model probabilities and validation-selected thresholds for the held-out test set.

## Test Classification, All Features

{test_metrics[['model_name', 'regime', 'n', 'threshold', 'accuracy', 'balanced_accuracy', 'auc', 'f1', 'up_day_rate']].round(4).to_markdown(index=False)}

## Test Long/Flat Backtest, All Features

{test_backtest[['model_name', 'threshold', 'strategy_cumulative_return', 'buy_hold_cumulative_return', 'strategy_sharpe', 'strategy_max_drawdown', 'exposure', 'trades']].round(4).to_markdown(index=False)}

## Robustness Snapshot

Robustness checks vary threshold and transaction costs. Thresholds are selected on validation for the main test report.

{best_robust[['split', 'model_name', 'threshold', 'transaction_cost', 'strategy_cumulative_return', 'strategy_sharpe', 'strategy_max_drawdown', 'exposure', 'trades']].round(4).to_markdown(index=False)}

## Continuous Return Modeling

{test_return[['model', 'feature_set', 'rmse', 'mae', 'r2', 'information_coefficient', 'direction_accuracy']].round(5).to_markdown(index=False)}

## Interpretation

- Balanced class weighting and calibrated probabilities make the models less dependent on raw accuracy.
- Thresholds are chosen on validation data, then applied to the test set.
- Predictive value remains weak; improvements should be framed as methodological cleanup rather than proof of a reliable trading edge.
"""
    (REPORT / "backtest_regime_summary.md").write_text(text, encoding="utf-8")


def main() -> None:
    preds = pd.read_csv(DATA / "model_predictions.csv", parse_dates=["Date"]).sort_values("Date")
    if "threshold" not in preds.columns:
        preds["threshold"] = 0.5

    regime_metrics = build_regime_metrics(preds)
    backtest_summary, backtest_regime, robustness = build_backtests(preds)
    failure_cases = build_failure_cases(preds)
    return_metrics, return_predictions = build_return_models()

    regime_metrics.to_csv(DATA / "model_regime_metrics.csv", index=False)
    backtest_summary.to_csv(DATA / "backtest_performance_summary.csv", index=False)
    backtest_regime.to_csv(DATA / "backtest_regime_summary.csv", index=False)
    robustness.to_csv(DATA / "robustness_threshold_cost_summary.csv", index=False)
    failure_cases.to_csv(DATA / "model_failure_cases.csv", index=False)
    return_metrics.to_csv(DATA / "return_model_metrics_summary.csv", index=False)
    return_predictions.to_csv(DATA / "return_model_predictions.csv", index=False)

    plot_backtest_equity(preds)
    plot_regime_returns(backtest_regime)
    plot_robustness(robustness)
    plot_return_predictions(return_predictions)
    write_summary(regime_metrics, backtest_summary, robustness, return_metrics)

    print("Evaluation complete.")
    print(f"Regime metrics: {DATA / 'model_regime_metrics.csv'}")
    print(f"Backtest summary: {DATA / 'backtest_performance_summary.csv'}")
    print(f"Report summary: {REPORT / 'backtest_regime_summary.md'}")


if __name__ == "__main__":
    main()
