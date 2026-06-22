"""
Author:Shubin Li

call fairness_evaluation and mitigation functions for nhanes dataset
"""

import pandas as pd
import numpy as np
from nhanes import models
from fairness_pipeline.fairness_eval import run_fairness_eval
import fairness_pipeline.mitigation as mitigation
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

RESULTS_DIR = Path(__file__).parent.parent / "results" / "nhanes"


# restore sensitive attributes from encoded feature matrix
def extract_sensitive_attrs(X: pd.DataFrame) -> pd.DataFrame:
    # gender  0 male,  1 female -> "Male" and "Female"
    sensitive_df = pd.DataFrame(index=X.index)
    sensitive_df["gender"] = X["gender"].map({0: "Male", 1: "Female"})

    # income_group ordinal ->  Low income / Near poverty / Above threshold -> 0 ,1 ,2
    # income_group derive from "INDFMPIR" Family income-to-poverty ratio, bins=[-np.inf, 1.3, 1.85, np.inf]
    income_map = {0: "Low income", 1: "Near poverty", 2: "Above threshold"}
    sensitive_df["income"] = X["income_group"].map(income_map)

    """
    race_map = {
        1: "Hispanic",
        2: "Hispanic",
        3: "White",
        4: "Black",
        6: "Asian",
        7: "Other Race",
    }
    Race category, Hispanic / White / Black / Asian / Other Race
    race_group -> one-hot to 4 col, exclude white
    race_group_Asian race_group_Black race_group_Hispanic race_group_Other Race
    """
    race_onehot_cols = [
        "race_group_Asian",
        "race_group_Black",
        "race_group_Hispanic",
        "race_group_Other Race",
    ]
    race_cols = (
        X[race_onehot_cols]
        .idxmax(axis=1)
        .where(X[race_onehot_cols].any(axis=1), "White")
    )
    sensitive_df["race"] = race_cols.str.replace("race_group_", "")

    return sensitive_df


def run_fairness_eval_for_all(
    fitted_models: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, dict[str, tuple[dict, pd.DataFrame]]]:
    """
    Run fairness evaluation for all sensitive attributes and all models
    
    """
    results = {}

    sensitive_df = extract_sensitive_attrs(X_test)

    for name, pipe in fitted_models.items():
        y_pred = pipe.predict(X_test)
        results[name] = {}
        for col in sensitive_df.columns:
            summary, details_df = run_fairness_eval(
                y_true=y_test, y_pred=y_pred, sensitive_col=sensitive_df[col]
            )
            results[name][col] = (summary, details_df)
    save_fairness_results(results)
    return results


def _get_model_performance(y_test: pd.Series, y_pred: pd.Series, y_proba=None) -> dict:

    row = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }

    if y_proba is not None:
        row["roc_auc"] = roc_auc_score(y_test, y_proba)
    else:
        row["roc_auc"] = np.NAN
    return row


def _get_model_fairness_eval(
    y_test: pd.Series, y_pred: pd.Series, sensitive_col: pd.Series
):
    fairness_summary, fairness_details_df = run_fairness_eval(
        y_true=y_test, y_pred=y_pred, sensitive_col=sensitive_col
    )
    return fairness_summary, fairness_details_df


# For supression
SENS_COL_DICT = {
    "gender": ["gender"],
    "income": ["income_group"],
    "race": [
        "race_group_Asian",
        "race_group_Black",
        "race_group_Hispanic",
        "race_group_Other Race",
    ],
}


# 3 models, 3 sensitive features, 4 mitigation method +  1 baseline, got 45 records.
def run_all_mitigations() -> list[dict]:
    """
    Run the full 3 models × 3 attrs × 4 mitigation grid.

    model | sensitive_attr | mitigation_method | performance_metrics | fairness_metrics
    """

    Baseline_models, X_train, X_test, y_train, y_test = (
        models.get_fitted_models_and_split_data()
    )

    # baseline fairness detail/summary, same models & split as mitigation grid
    run_fairness_eval_for_all(Baseline_models, X_test, y_test)

    # sensitive_train_df for mitigation
    # sensitive_test_df for fairness evaluation
    sensitive_train_df = extract_sensitive_attrs(X_train)
    sensitive_test_df = extract_sensitive_attrs(X_test)

    all_results = []

    for name, pipe in Baseline_models.items():
        y_pred_base = pipe.predict(X_test)
        y_proba_base = pipe.predict_proba(X_test)[:, 1]
        performance_base = _get_model_performance(y_test, y_pred_base, y_proba_base)

        for col in sensitive_train_df.columns:
            sens_train = sensitive_train_df[col]
            sens_test = sensitive_test_df[col]
            fairness_summary_base, fairness_details_base = _get_model_fairness_eval(
                y_test, y_pred_base, sens_test
            )
            # 1. Baseline model performance and fairness evaluation
            all_results.append(
                _build_row(
                    name, col, "Baseline", performance_base, fairness_summary_base
                )
            )

            # 2. reweighing
            print(f" >>>> {name} | {col} | Reweighing ...")
            rw_model = mitigation.fit_model_with_reweighing(
                pipe, X_train, y_train, sens_train
            )

            y_pred_rw = rw_model.predict(X_test)
            y_proba_rw = rw_model.predict_proba(X_test)[:, 1]
            performance_rw = _get_model_performance(y_test, y_pred_rw, y_proba_rw)
            fairness_summary_rw, _ = _get_model_fairness_eval(
                y_test, y_pred_rw, sens_test
            )
            all_results.append(
                _build_row(name, col, "Reweighing", performance_rw, fairness_summary_rw)
            )

            # 3. ExponentiatedGradient
            print(f" >>>> {name} | {col} | ExponentiatedGradient ...")
            exg_model = mitigation.apply_exponentiated_gradient(
                pipe, X_train, y_train, sens_train
            )

            # exponentiatedGradient don't support predict_proba
            y_pred_exg = exg_model.predict(X_test, random_state=42)

            performance_exg = _get_model_performance(y_test, y_pred_exg)
            fairness_summary_exg, _ = _get_model_fairness_eval(
                y_test, y_pred_exg, sens_test
            )
            all_results.append(
                _build_row(
                    name,
                    col,
                    "ExponentiatedGradient",
                    performance_exg,
                    fairness_summary_exg,
                )
            )

            # 4. ThresholdOptimizer
            print(f" >>>> {name} | {col} | ThresholdOptimizer ...")
            to_model = mitigation.apply_threshold_optimizer(
                pipe, X_train, y_train, sens_train
            )

            y_pred_to = to_model.predict(
                X_test, random_state=42, sensitive_features=sens_test
            )
            # thresholdOptimizer do nothing with probability, but threshold, so probability same as baseline
            performance_to = _get_model_performance(y_test, y_pred_to)
            fairness_summary_to, _ = _get_model_fairness_eval(
                y_test, y_pred_to, sens_test
            )

            all_results.append(
                _build_row(
                    name,
                    col,
                    "ThresholdOptimizer",
                    performance_to,
                    fairness_summary_to,
                )
            )

            # 5 Suppression
            print(f" >>>> {name} | {col} | Suppression ...")
            sup_model, keep_cols = mitigation.apply_suppression(
                pipe, X_train, y_train, SENS_COL_DICT[col]
            )

            X_test_reduced = X_test[keep_cols]
            y_pred_sup = sup_model.predict(X_test_reduced)
            y_proba_sup = sup_model.predict_proba(X_test_reduced)[:, 1]
            performance_sup = _get_model_performance(y_test, y_pred_sup, y_proba_sup)
            fairness_summary_sup, _ = _get_model_fairness_eval(
                y_test, y_pred_sup, sens_test
            )
            all_results.append(
                _build_row(
                    name, col, "Suppression", performance_sup, fairness_summary_sup
                )
            )
    save_mitigation_results(pd.DataFrame(all_results))
    return all_results


def _build_row(
    model_name,
    sensitive_attr,
    mitigation_method,
    performance: dict,
    fairness_summary: dict,
) -> dict:
    row = {
        "model": model_name,
        "sensitive_attr": sensitive_attr,
        "miti_method": mitigation_method,
        **performance,
        **fairness_summary,
    }
    return row


def save_mitigation_results(df: pd.DataFrame):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "nhanes_mitigation_results.csv"
    df.to_csv(path, index=False, float_format="%.4f")
    print(f"\nMitigation results saved -> {path}")
    print(df.to_string(index=False))


"""
Save fairness evaluation results to disk.

Two outputs:
  1. detail CSV  — per-group metrics for every (model, sensitive_attr) pair
  2. summary CSV — one row per (model, sensitive_attr), three fairness metrics values only
"""


def save_fairness_results(
    all_results: dict[str, dict[str, tuple[dict, pd.DataFrame]]],
    results_dir: Path = RESULTS_DIR,
):

    detail_rows = []
    summary_rows = []
    results_dir.mkdir(parents=True, exist_ok=True)
    for model_name, attr_dict in all_results.items():
        for attr_name, (summary, details_df) in attr_dict.items():

            #  detail: flatten per-group rows
            for group_name, row in details_df.iterrows():
                detail_rows.append(
                    {
                        "model": model_name,
                        "sensitive_attr": attr_name,
                        "group": group_name,
                        **row.to_dict(),
                    }
                )

            #  summary: one row per (model, attr)
            summary_rows.append(
                {
                    "model": model_name,
                    "sensitive_attr": attr_name,
                    **summary,
                }
            )

    detail_df = pd.DataFrame(detail_rows)
    summary_df = pd.DataFrame(summary_rows)

    detail_path = results_dir / "nhanes_fairness_baseline_detail.csv"
    summary_path = results_dir / "nhanes_fairness_baseline_summary.csv"

    detail_df.to_csv(detail_path, index=False, float_format="%.4f")
    summary_df.to_csv(summary_path, index=False, float_format="%.4f")

    print(f"Detail  saved -> {detail_path}")
    print(f"Summary saved -> {summary_path}")

    print(detail_df)
    print(summary_df)
    return detail_df, summary_df


if __name__ == "__main__":

    # x = run_fairness_eval_for_all()
    run_all_mitigations()
    
