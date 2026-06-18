"""
Shubin Li

call fairness_evaluation functions for nhanes dataset
"""

import pandas as pd
from nhanes import models
from fairness_pipeline.fairness_eval import run_fairness_eval
from pathlib import Path


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


def run_fairness_eval_for_all() -> dict[str, dict[str, tuple[dict, pd.DataFrame]]]:
    """
    Run fairness evaluation for all sensitive attributes and all models
    """
    results = {}
    fitted_models, X_train, X_test, y_train, y_test = (
        models.get_fitted_models_and_split_data()
    )
    sensitive_df = extract_sensitive_attrs(X_test)

    for name, pipe in fitted_models.items():
        y_pred = pipe.predict(X_test)
        results[name] = {}
        for col in sensitive_df.columns:
            summary, details_df = run_fairness_eval(
                y_true=y_test, y_pred=y_pred, sensitive_col=sensitive_df[col]
            )
            results[name][col] = (summary, details_df)

    return results





RESULTS_DIR = Path(__file__).parent.parent / "results" / "nhanes"

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

    x = run_fairness_eval_for_all()
    save_fairness_results(x)