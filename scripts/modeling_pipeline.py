"""Train calibrated direction models and write modeling artifacts.

This script is the reproducible version of notebook 04. It uses only past data
for feature lags, tunes hyperparameters and classification thresholds on the
validation split, and applies the selected thresholds to the held-out test split.
"""

from __future__ import annotations

import warnings
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = PROJECT_ROOT / ".cache"
CACHE_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


warnings.filterwarnings("ignore")

DATA = PROJECT_ROOT / "data" / "processed"
FIG = PROJECT_ROOT / "figures"
FIG.mkdir(exist_ok=True)

RANDOM_STATE = 42
THRESHOLD_GRID = np.round(np.arange(0.35, 0.651, 0.01), 2)
XGB_PARAM_GRID = [
    {"max_depth": max_depth, "learning_rate": learning_rate, "n_estimators": n_estimators}
    for max_depth in [2, 3]
    for learning_rate in [0.03, 0.07]
    for n_estimators in [150, 300]
]


def add_lags(df: pd.DataFrame, cols: list[str], lags: tuple[int, ...] = (1, 2, 5)) -> None:
    for col in cols:
        if col not in df.columns:
            continue
        for lag in lags:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)


def load_and_engineer_features() -> tuple[pd.DataFrame, dict[str, list[str]], bool]:
    v2_path = DATA / "daily_with_sentiment_v2.csv"
    v1_path = DATA / "daily_with_sentiment.csv"
    src = v2_path if v2_path.exists() else v1_path

    df = pd.read_csv(src, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    has_v2 = "finbert_score_mean" in df.columns

    dict_raw = [
        "avg_finance_sentiment",
        "net_positive_negative_share",
        "positive_headline_share",
        "negative_headline_share",
        "finance_sentiment_std",
    ]
    general_raw = ["avg_general_sentiment"]
    vader_raw = ["vader_compound_mean", "vader_pos_share", "vader_neg_share"] if has_v2 else []
    finbert_raw = ["finbert_score_mean", "finbert_pos_share", "finbert_neg_share"] if has_v2 else []
    topic_cols = [f"topic_{k}" for k in range(8)] if has_v2 and "topic_0" in df.columns else []

    add_lags(df, dict_raw + general_raw + vader_raw + finbert_raw)

    for col in ["avg_finance_sentiment", "vader_compound_mean", "finbert_score_mean"]:
        if col in df.columns:
            df[f"{col}_roll5"] = df[col].shift(1).rolling(5).mean()
            df[f"{col}_roll10"] = df[col].shift(1).rolling(10).mean()

    macro_cols = [
        "vix",
        "term_spread_10y_2y",
        "cpi_yoy",
        "unemployment_rate",
        "recession_indicator",
        "fed_funds_rate",
    ]
    df["headline_count_log"] = np.log1p(df["headline_count"])
    df["fin_sent_x_vix"] = df["avg_finance_sentiment"] * df["vix"]
    if has_v2:
        df["finbert_x_vix"] = df["finbert_score_mean"] * df["vix"]

    lag_roll_cols = [col for col in df.columns if "lag" in col or "roll" in col]
    df_model = df.dropna(subset=lag_roll_cols).reset_index(drop=True)

    def lag_cols(prefixes: list[str]) -> list[str]:
        return [
            col
            for prefix in prefixes
            for col in df_model.columns
            if col.startswith(prefix) and ("lag" in col or "roll" in col)
        ]

    features_dict = lag_cols(dict_raw) + ["headline_count_log"]
    features_general = lag_cols(general_raw) + ["headline_count_log"]
    features_vader = lag_cols(["vader_"]) + ["headline_count_log"] if has_v2 else []
    features_finbert = lag_cols(["finbert_"]) + ["headline_count_log"] if has_v2 else []
    features_topics = topic_cols + ["headline_count_log"]
    features_all = sorted(
        set(
            features_dict
            + features_general
            + features_vader
            + features_finbert
            + features_topics
            + macro_cols
            + ["fin_sent_x_vix"]
            + (["finbert_x_vix"] if has_v2 else [])
        )
    )

    feature_sets = {
        "dict": features_dict,
        "general": features_general,
        "vader": features_vader,
        "finbert": features_finbert,
        "all": features_all,
    }
    feature_sets = {name: cols for name, cols in feature_sets.items() if cols}
    return df_model, feature_sets, has_v2


def split_by_date(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = df[df["Date"] < "2020-01-01"].copy()
    val = df[(df["Date"] >= "2020-01-01") & (df["Date"] < "2022-01-01")].copy()
    test = df[df["Date"] >= "2022-01-01"].copy()
    return train, val, test


def safe_auc(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    if pd.Series(y_true).nunique() < 2:
        return np.nan
    return float(roc_auc_score(y_true, y_proba))


def metrics_dict(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    threshold: float | None = None,
) -> dict[str, float]:
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_proba is not None:
        out["auc"] = safe_auc(y_true, y_proba)
        out["brier"] = brier_score_loss(y_true, y_proba)
    if threshold is not None:
        out["threshold"] = threshold
    return out


def choose_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> tuple[float, pd.DataFrame]:
    rows = []
    for threshold in THRESHOLD_GRID:
        pred = (y_proba >= threshold).astype(int)
        row = metrics_dict(y_true, pred, y_proba, float(threshold))
        row["threshold_distance_from_0_5"] = abs(float(threshold) - 0.5)
        rows.append(row)

    table = pd.DataFrame(rows)
    best = table.sort_values(
        ["balanced_accuracy", "f1", "threshold_distance_from_0_5"],
        ascending=[False, False, True],
    ).iloc[0]
    return float(best["threshold"]), table


def positive_class_weight(y: np.ndarray) -> float:
    positives = max(int((y == 1).sum()), 1)
    negatives = int((y == 0).sum())
    return negatives / positives


def xgb_classifier(params: dict[str, float], scale_pos_weight: float) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=int(params["n_estimators"]),
        max_depth=int(params["max_depth"]),
        learning_rate=float(params["learning_rate"]),
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
    )


def fit_logit(
    feature_set: str,
    feats: list[str],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[dict, list[dict], dict]:
    scaler = StandardScaler().fit(train_df[feats])
    x_train = scaler.transform(train_df[feats])
    x_val = scaler.transform(val_df[feats])
    x_test = scaler.transform(test_df[feats])
    y_train = train_df["direction_next_day"].values
    y_val = val_df["direction_next_day"].values
    y_test = test_df["direction_next_day"].values

    base = LogisticRegression(
        max_iter=2000,
        C=1.0,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    clf = CalibratedClassifierCV(estimator=base, method="sigmoid", cv=3)
    clf.fit(x_train, y_train)

    coef_model = LogisticRegression(
        max_iter=2000,
        C=1.0,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    ).fit(x_train, y_train)
    coef_table = pd.DataFrame({"feature": feats, "coef": coef_model.coef_[0]})
    try:
        sm_res = sm.Logit(y_train, sm.add_constant(x_train)).fit(disp=0, maxiter=300)
        p_values = pd.Series(sm_res.pvalues[1:], index=feats)
        coef_table["p_value"] = coef_table["feature"].map(p_values)
    except Exception:
        coef_table["p_value"] = np.nan

    val_proba = clf.predict_proba(x_val)[:, 1]
    threshold, threshold_table = choose_threshold(y_val, val_proba)

    rows = []
    for split_name, x_split, y_split in [("val", x_val, y_val), ("test", x_test, y_test)]:
        proba = clf.predict_proba(x_split)[:, 1]
        pred = (proba >= threshold).astype(int)
        row = metrics_dict(y_split, pred, proba, threshold)
        row.update({"model": "Logit", "feature_set": feature_set, "split": split_name})
        rows.append(row)

    threshold_record = {
        "model": "Logit",
        "feature_set": feature_set,
        "selected_threshold": threshold,
        "threshold_objective": "validation_balanced_accuracy",
        "val_balanced_accuracy": float(threshold_table["balanced_accuracy"].max()),
        "val_f1_at_selected_threshold": float(
            threshold_table.loc[threshold_table["threshold"] == threshold, "f1"].iloc[0]
        ),
    }
    model = {
        "clf": clf,
        "scaler": scaler,
        "feats": feats,
        "coef": coef_table,
        "threshold": threshold,
    }
    return model, rows, threshold_record


def fit_xgb(
    feature_set: str,
    feats: list[str],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[dict, list[dict], dict, pd.DataFrame]:
    x_train = train_df[feats]
    x_val = val_df[feats]
    x_test = test_df[feats]
    y_train = train_df["direction_next_day"].values
    y_val = val_df["direction_next_day"].values
    y_test = test_df["direction_next_day"].values
    scale_weight = positive_class_weight(y_train)

    tuning_rows = []
    for params in XGB_PARAM_GRID:
        candidate = xgb_classifier(params, scale_weight)
        candidate.fit(x_train, y_train)
        val_proba = candidate.predict_proba(x_val)[:, 1]
        val_pred_05 = (val_proba >= 0.5).astype(int)
        tuning_rows.append(
            {
                "model": "XGB",
                "feature_set": feature_set,
                **params,
                "scale_pos_weight": scale_weight,
                "val_auc": safe_auc(y_val, val_proba),
                "val_balanced_accuracy_05": balanced_accuracy_score(y_val, val_pred_05),
                "val_brier": brier_score_loss(y_val, val_proba),
            }
        )

    tuning_df = pd.DataFrame(tuning_rows)
    best_row = tuning_df.sort_values(
        ["val_auc", "val_balanced_accuracy_05", "val_brier"],
        ascending=[False, False, True],
    ).iloc[0]
    best_params = {
        "max_depth": int(best_row["max_depth"]),
        "learning_rate": float(best_row["learning_rate"]),
        "n_estimators": int(best_row["n_estimators"]),
    }

    base = xgb_classifier(best_params, scale_weight)
    clf = CalibratedClassifierCV(estimator=base, method="sigmoid", cv=3)
    clf.fit(x_train, y_train)

    importance_model = xgb_classifier(best_params, scale_weight)
    importance_model.fit(x_train, y_train)

    val_proba = clf.predict_proba(x_val)[:, 1]
    threshold, threshold_table = choose_threshold(y_val, val_proba)

    rows = []
    for split_name, x_split, y_split in [("val", x_val, y_val), ("test", x_test, y_test)]:
        proba = clf.predict_proba(x_split)[:, 1]
        pred = (proba >= threshold).astype(int)
        row = metrics_dict(y_split, pred, proba, threshold)
        row.update({"model": "XGB", "feature_set": feature_set, "split": split_name})
        rows.append(row)

    threshold_record = {
        "model": "XGB",
        "feature_set": feature_set,
        "selected_threshold": threshold,
        "threshold_objective": "validation_balanced_accuracy",
        "val_balanced_accuracy": float(threshold_table["balanced_accuracy"].max()),
        "val_f1_at_selected_threshold": float(
            threshold_table.loc[threshold_table["threshold"] == threshold, "f1"].iloc[0]
        ),
    }
    model = {
        "clf": clf,
        "feats": feats,
        "threshold": threshold,
        "params": best_params,
        "scale_pos_weight": scale_weight,
        "importance_model": importance_model,
    }
    return model, rows, threshold_record, tuning_df


def baseline_rows(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> list[dict]:
    rows = []
    train_up_rate = train_df["direction_next_day"].mean()
    majority_class = int(train_up_rate >= 0.5)
    majority_proba = np.full(len(test_df), train_up_rate, dtype=float)

    for split_name, split_df in [("val", val_df), ("test", test_df)]:
        y_true = split_df["direction_next_day"].values
        majority_pred = np.full(len(split_df), majority_class, dtype=int)
        majority_proba = np.full(len(split_df), train_up_rate, dtype=float)
        row = metrics_dict(y_true, majority_pred, majority_proba, 0.5)
        row.update({"model": "Baseline_majority", "feature_set": "-", "split": split_name})
        rows.append(row)

        momentum_pred = (split_df["return_same_day"].fillna(0) > 0).astype(int).values
        momentum_proba = momentum_pred.astype(float)
        row = metrics_dict(y_true, momentum_pred, momentum_proba, 0.5)
        row.update({"model": "Baseline_momentum", "feature_set": "-", "split": split_name})
        rows.append(row)

    return rows


def collect_predictions(
    model_name: str,
    feature_set: str,
    model: dict,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> pd.DataFrame:
    parts = []
    threshold = float(model["threshold"])
    for split_name, split_df in [("val", val_df), ("test", test_df)]:
        feats = model["feats"]
        x_split = split_df[feats]
        if model_name == "Logit":
            x_split = model["scaler"].transform(x_split)
        proba = model["clf"].predict_proba(x_split)[:, 1]
        pred = (proba >= threshold).astype(int)
        parts.append(
            pd.DataFrame(
                {
                    "Date": split_df["Date"].values,
                    "regime": split_df["regime"].values,
                    "y_true": split_df["direction_next_day"].values,
                    "return_next_day": split_df["return_next_day"].values,
                    "y_pred": pred,
                    "y_pred_proba": proba,
                    "model_name": model_name,
                    "feature_set": feature_set,
                    "split": split_name,
                    "threshold": threshold,
                    "threshold_source": "validation_balanced_accuracy",
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


def fit_expanding_model(
    model_name: str,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    params: dict | None = None,
) -> tuple[object, StandardScaler | None]:
    if model_name == "Logit":
        scaler = StandardScaler().fit(x_train)
        x_train_scaled = scaler.transform(x_train)
        base = LogisticRegression(
            max_iter=2000,
            C=1.0,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )
        clf = CalibratedClassifierCV(estimator=base, method="sigmoid", cv=3)
        clf.fit(x_train_scaled, y_train)
        return clf, scaler

    scale_weight = positive_class_weight(y_train)
    base = xgb_classifier(params or {"max_depth": 2, "learning_rate": 0.07, "n_estimators": 150}, scale_weight)
    clf = CalibratedClassifierCV(estimator=base, method="sigmoid", cv=3)
    clf.fit(x_train, y_train)
    return clf, None


def expanding_window_predict(
    df_model: pd.DataFrame,
    feats: list[str],
    model_name: str,
    feature_set: str,
    params: dict | None = None,
    start_year: int = 2009,
) -> pd.DataFrame:
    parts = []
    for year in range(start_year, int(df_model["year"].max()) + 1):
        train = df_model[df_model["year"] < year]
        test = df_model[df_model["year"] == year]
        if len(train) < 80 or len(test) < 20:
            continue

        x_train = train[feats]
        y_train = train["direction_next_day"].values
        x_test = test[feats]
        clf, scaler = fit_expanding_model(model_name, x_train, y_train, params=params)
        x_predict = scaler.transform(x_test) if scaler is not None else x_test
        proba = clf.predict_proba(x_predict)[:, 1]
        threshold = 0.5
        parts.append(
            pd.DataFrame(
                {
                    "Date": test["Date"].values,
                    "regime": test["regime"].values,
                    "y_true": test["direction_next_day"].values,
                    "return_next_day": test["return_next_day"].values,
                    "y_pred": (proba >= threshold).astype(int),
                    "y_pred_proba": proba,
                    "model_name": model_name,
                    "feature_set": feature_set,
                    "split": "oos_expanding",
                    "threshold": threshold,
                    "threshold_source": "expanding_default_0.50",
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


def plot_split_timeline(df_model: pd.DataFrame, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 2.5))
    ax.plot(df_model["Date"], df_model["CP"], color="black", lw=0.6)
    ax.axvspan(train_df.Date.min(), train_df.Date.max(), alpha=0.15, color="steelblue", label="Train")
    ax.axvspan(val_df.Date.min(), val_df.Date.max(), alpha=0.20, color="orange", label="Val")
    ax.axvspan(test_df.Date.min(), test_df.Date.max(), alpha=0.20, color="green", label="Test")
    ax.set_title("Time-aware Split - S&P 500 Close")
    ax.set_ylabel("S&P 500")
    ax.legend(loc="upper left")
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    plt.tight_layout()
    plt.savefig(FIG / "fig1_split_timeline.png", dpi=130)
    plt.close(fig)


def plot_method_comparison(results_df: pd.DataFrame) -> None:
    test_metrics = results_df[results_df["split"] == "test"].copy()
    order = [x for x in ["dict", "general", "vader", "finbert", "all"] if x in test_metrics["feature_set"].unique()]
    x = np.arange(len(order))
    width = 0.36

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
    for ax, metric, title, baseline in [
        (axes[0], "auc", "Test AUC", 0.5),
        (axes[1], "balanced_accuracy", "Test Balanced Accuracy", 0.5),
    ]:
        logit_vals = [
            test_metrics[(test_metrics.model == "Logit") & (test_metrics.feature_set == name)][metric].iloc[0]
            for name in order
        ]
        xgb_vals = [
            test_metrics[(test_metrics.model == "XGB") & (test_metrics.feature_set == name)][metric].iloc[0]
            for name in order
        ]
        ax.bar(x - width / 2, logit_vals, width, label="Logit", color="steelblue")
        ax.bar(x + width / 2, xgb_vals, width, label="XGB", color="darkorange")
        ax.axhline(baseline, color="red", ls="--", lw=0.8)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(order, rotation=20)
        ax.set_ylim(0.35, 0.65)
        for i, (left, right) in enumerate(zip(logit_vals, xgb_vals)):
            ax.text(i - width / 2, left + 0.004, f"{left:.3f}", ha="center", fontsize=8)
            ax.text(i + width / 2, right + 0.004, f"{right:.3f}", ha="center", fontsize=8)
    axes[0].legend(loc="lower left", fontsize=8)
    fig.suptitle("Calibrated Models with Validation-Tuned Thresholds")
    plt.tight_layout()
    plt.savefig(FIG / "fig2_sentiment_method_comparison.png", dpi=130)
    plt.close(fig)


def plot_logit_coefficients(logit_models: dict[str, dict]) -> None:
    target_fs = "finbert" if "finbert" in logit_models else "dict"
    coef_table = logit_models[target_fs]["coef"].copy().sort_values("coef")
    colors = ["tab:red" if pd.notna(p) and p < 0.05 else "lightgray" for p in coef_table["p_value"]]
    fig, ax = plt.subplots(figsize=(8, max(4, len(coef_table) * 0.25)))
    ax.barh(coef_table["feature"], coef_table["coef"], color=colors)
    ax.axvline(0, color="black", lw=0.5)
    ax.set_title(f"Balanced Logit Coefficients - {target_fs} (red = p<0.05)")
    ax.set_xlabel("Standardized coefficient")
    plt.tight_layout()
    plt.savefig(FIG / "fig3_logit_coefficients.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_xgb_importance(xgb_models: dict[str, dict]) -> None:
    model = xgb_models["all"]["importance_model"]
    feats = xgb_models["all"]["feats"]
    imp = pd.DataFrame({"feature": feats, "importance": model.feature_importances_})
    imp = imp.sort_values("importance", ascending=True).tail(15)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(imp["feature"], imp["importance"], color="steelblue")
    ax.set_title("XGBoost Top-15 Feature Importance (all features)")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig(FIG / "fig4_xgb_feature_importance.png", dpi=130)
    plt.close(fig)


def plot_roc(test_df: pd.DataFrame, logit_models: dict[str, dict], xgb_models: dict[str, dict]) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    plot_targets = []
    for feature_set in ["general", "dict", "vader", "finbert", "all"]:
        if feature_set in logit_models:
            plot_targets.append(("Logit", feature_set, logit_models[feature_set]))
    plot_targets.append(("XGB", "all", xgb_models["all"]))

    for model_name, feature_set, model in plot_targets:
        feats = model["feats"]
        x_test = test_df[feats]
        if model_name == "Logit":
            x_test = model["scaler"].transform(x_test)
        proba = model["clf"].predict_proba(x_test)[:, 1]
        fpr, tpr, _ = roc_curve(test_df["direction_next_day"], proba)
        auc_value = safe_auc(test_df["direction_next_day"].values, proba)
        ax.plot(fpr, tpr, label=f"{model_name}-{feature_set} (AUC={auc_value:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.6)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC - Test Set (2022-01 to 2024-03)")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG / "fig5_roc_test.png", dpi=130)
    plt.close(fig)


def main() -> None:
    np.random.seed(RANDOM_STATE)
    df_model, feature_sets, has_v2 = load_and_engineer_features()
    train_df, val_df, test_df = split_by_date(df_model)

    print(f"Using v2 sentiment features: {has_v2}")
    print(f"Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")
    print("Feature set sizes:", {name: len(cols) for name, cols in feature_sets.items()})

    results = baseline_rows(train_df, val_df, test_df)
    thresholds = []
    tuning_tables = []
    logit_models: dict[str, dict] = {}
    xgb_models: dict[str, dict] = {}

    for feature_set, feats in feature_sets.items():
        model, rows, threshold_record = fit_logit(feature_set, feats, train_df, val_df, test_df)
        logit_models[feature_set] = model
        results.extend(rows)
        thresholds.append(threshold_record)

    for feature_set, feats in feature_sets.items():
        model, rows, threshold_record, tuning_df = fit_xgb(feature_set, feats, train_df, val_df, test_df)
        xgb_models[feature_set] = model
        results.extend(rows)
        thresholds.append(threshold_record)
        tuning_tables.append(tuning_df)

    prediction_parts = []
    for feature_set, model in logit_models.items():
        prediction_parts.append(collect_predictions("Logit", feature_set, model, val_df, test_df))
    for feature_set, model in xgb_models.items():
        prediction_parts.append(collect_predictions("XGB", feature_set, model, val_df, test_df))

    prediction_parts.append(
        expanding_window_predict(df_model, feature_sets["all"], "Logit", "all", start_year=2009)
    )
    prediction_parts.append(
        expanding_window_predict(
            df_model,
            feature_sets["all"],
            "XGB",
            "all",
            params=xgb_models["all"]["params"],
            start_year=2009,
        )
    )

    results_df = pd.DataFrame(results)
    predictions = pd.concat(prediction_parts, ignore_index=True)
    thresholds_df = pd.DataFrame(thresholds)
    tuning_df = pd.concat(tuning_tables, ignore_index=True)

    results_df.to_csv(DATA / "model_metrics_summary.csv", index=False)
    predictions.to_csv(DATA / "model_predictions.csv", index=False)
    thresholds_df.to_csv(DATA / "model_thresholds.csv", index=False)
    tuning_df.to_csv(DATA / "model_tuning_summary.csv", index=False)

    plot_split_timeline(df_model, train_df, val_df, test_df)
    plot_method_comparison(results_df)
    plot_logit_coefficients(logit_models)
    plot_xgb_importance(xgb_models)
    plot_roc(test_df, logit_models, xgb_models)

    print("Saved model metrics, predictions, thresholds, tuning summary, and figures 1-5.")
    print(
        results_df[results_df["split"] == "test"]
        .sort_values("balanced_accuracy", ascending=False)
        [["model", "feature_set", "threshold", "accuracy", "balanced_accuracy", "f1", "auc", "brier"]]
        .round(4)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
