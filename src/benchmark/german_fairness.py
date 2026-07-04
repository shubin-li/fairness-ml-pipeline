"""
Author: Shubin Li

Standalone German Credit fairness-mitigation benchmark.

Self-contained cross-dataset benchmark that reuses ONLY the dataset-agnostic
common modules (fairness_pipeline.mitigation, fairness_pipeline.fairness_eval).

Same evaluation protocol as NHANES and Adult:
  * 5-fold stratified CV on the train split (reporting only)
  * 20% held-out stratified test split
  * 3 models (LR / RF / XGB) x 4 mitigations
    (Reweighing / ExponentiatedGradient / ThresholdOptimizer / Suppression)
    + Baseline

Sensitive attributes: sex and age_group.
Target: good credit (original label 1) as the positive class, bad credit
(original label 2) as negative. Positive rate is about 70 percent, so here the
positive class is the majority and the negative class is the scarce one.

Outputs (schema-identical to the NHANES and Adult result CSVs) into
results/benchmark/german/:
  * german_baseline_cv.csv
  * german_baseline_test.csv
  * german_fairness_baseline_detail.csv
  * german_fairness_baseline_summary.csv
  * german_mitigation_results.csv   (30 rows)
  * german_mitigation_detail.csv

"""

import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path

from ucimlrepo import fetch_ucirepo

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    StratifiedKFold,
    train_test_split,
    cross_validate,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

# only the two dataset-agnostic common modules
from fairness_pipeline.fairness_eval import run_fairness_eval
import fairness_pipeline.mitigation as mitigation


RANDOM_STATE = 42
TEST_SIZE = 0.2
N_FOLDS = 5

RESULTS_DIR = Path(__file__).parent.parent / "results" / "benchmark" / "german"

# UCI Statlog German Credit Data (id 144). Attribute9 encodes personal status
# and sex; Attribute13 is age in years. Numerical columns get standardized,
# every other column is one-hot encoded, mirroring adult_fairness.
GERMAN_UCI_ID = 144
SEX_COL = "Attribute9"
AGE_COL = "Attribute13"
NUMERIC_COLS = [
    "Attribute2",
    "Attribute5",
    "Attribute8",
    "Attribute11",
    "Attribute13",
    "Attribute16",
    "Attribute18",
]
# A91/A93/A94 are male, A92/A95 are female (A95 absent in this copy)
SEX_MAP = {
    "A91": "Male",
    "A92": "Female",
    "A93": "Male",
    "A94": "Male",
    "A95": "Female",
}
AGE_THRESHOLD = 25  # age < 25 -> young, age >= 25 -> old


# =====================================================================
# 1. Data loading + encoding
# =====================================================================
def load_german() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Load the full German Credit dataset and encode it into a fully-numeric
    feature matrix, mirroring the Adult convention so that the sensitive
    attributes stay recoverable from the encoded matrix:

      * sex       -> single column `sex` (0 = Male, 1 = Female)
      * age_group -> single column `age_group` (0 = old, 1 = young)
      * every other categorical -> one-hot (dtype int)
      * numerical columns -> passed through, scaled later inside the pipeline

    Returns (X_encoded, y, raw_df) where raw_df is the un-encoded frame kept
    only for profiling and EDA reporting.
    """
    german = fetch_ucirepo(id=GERMAN_UCI_ID)
    X_raw = german.data.features.copy()
    y_raw = german.data.targets.copy()

    # target: good credit (label 1) is the positive class (binary 1/0)
    target_col = y_raw.columns[0]
    y = (y_raw[target_col].astype(int) == 1).astype(int)
    y.name = "good_credit"

    # readable sensitive attributes for profiling
    raw_df = X_raw.copy()
    raw_df["sex"] = X_raw[SEX_COL].map(SEX_MAP)
    raw_df["age_group"] = np.where(X_raw[AGE_COL] < AGE_THRESHOLD, "young", "old")
    raw_df["good_credit"] = y.values

    # ---- encode into numeric matrix ----
    df = X_raw.copy()

    # sex -> single binary column, Male=0 / Female=1
    df["sex"] = (df[SEX_COL].map(SEX_MAP) == "Female").astype(int)
    df = df.drop(columns=[SEX_COL])

    # age_group -> single binary column, old=0 / young=1; keep raw age as a feature
    df["age_group"] = (df[AGE_COL] < AGE_THRESHOLD).astype(int)

    # every remaining object column to one-hot
    cat_cols = [
        c
        for c in df.columns
        if str(df[c].dtype) == "category" or df[c].dtype == object
    ]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=False, dtype=int)

    df = df.reset_index(drop=True)
    y = y.reset_index(drop=True)

    return df, y, raw_df


# recover human-readable sensitive attributes from the encoded matrix,
# same role as adult_fairness.extract_sensitive_attrs (German-specific)
def extract_sensitive_attrs(X: pd.DataFrame) -> pd.DataFrame:
    sensitive_df = pd.DataFrame(index=X.index)

    # sex: 0 male, 1 female
    sensitive_df["sex"] = X["sex"].map({0: "Male", 1: "Female"})

    # age_group: 0 old, 1 young
    sensitive_df["age_group"] = X["age_group"].map({0: "old", 1: "young"})

    return sensitive_df


# Suppression drop-cols: drop the sensitive column(s) and their direct proxies.
#   sex       -> `sex`
#   age_group -> `age_group` + the raw age column (Attribute13)
def _suppression_cols(X: pd.DataFrame) -> dict[str, list[str]]:
    age_cols = ["age_group"] + [c for c in X.columns if c == AGE_COL]
    return {
        "sex": ["sex"],
        "age_group": age_cols,
    }


# =====================================================================
# 2. Profiling + EDA
# =====================================================================
def dataset_profiler(df: pd.DataFrame, target: str, sens_cols: list[str]) -> None:
    """
    Place German Credit on the three axes of the characteristic spectrum
    (class imbalance, sensitive-attr cardinality, true-prevalence gap) and
    flag small unstable groups. German has only 1000 rows, so age_group `young`
    can be small; a group with count < 50 or fewer than 15 true positives is
    marked unstable.
    """
    n = len(df)
    pos_rate = df[target].mean()

    print("=" * 60)
    print("PROFILE — German Credit")
    print("=" * 60)
    print(f"total samples      : {n}")
    print(f"features           : {df.shape[1] - 1}")
    print(
        f"positive class rate: {pos_rate:.4f}  "
        f"(imbalance {pos_rate:.1%} pos / {1 - pos_rate:.1%} neg; "
        f"positive is the majority class)"
    )

    print("\nsensitive attributes (cardinality / group sizes / prevalence):")
    for col in sens_cols:
        groups = df[col].value_counts()
        prevalence = df.groupby(col, observed=True)[target].mean()
        gap = prevalence.max() - prevalence.min()
        print(f"\n  [{col}]  cardinality = {df[col].nunique()}")
        for grp in groups.index:
            cnt = int(groups[grp])
            prev = prevalence[grp]
            true_pos = int(df[df[col] == grp][target].sum())
            unstable = cnt < 50 or true_pos < 15
            flag = "  <== UNSTABLE (small group)" if unstable else ""
            print(
                f"    {str(grp):<10} n={cnt:>4} ({cnt / n:5.1%})  "
                f"true_prevalence={prev:.4f}  true_positives={true_pos:>4}{flag}"
            )
        print(f"    -> prevalence gap (max-min) = {gap:.4f}")

    print("\n" + "-" * 60)
    print("spectrum position:")
    print(f"  imbalance  (pos rate)          : {pos_rate:.3f}")
    card = {c: df[c].nunique() for c in sens_cols}
    print(f"  cardinality (per sens attr)    : {card}")
    prev_gaps = {}
    for col in sens_cols:
        prevalence = df.groupby(col, observed=True)[target].mean()
        prev_gaps[col] = round(prevalence.max() - prevalence.min(), 4)
    print(f"  prevalence gap (per sens attr) : {prev_gaps}")
    print("=" * 60 + "\n")


# =====================================================================
# 3. Models (same 3 models as NHANES / Adult: LR / RF / XGB)
# =====================================================================
def get3_pipe_models(y_train: pd.Series) -> dict[str, Pipeline]:
    """
    Same three models and imbalance handling strategy as the Adult pipeline:
      LR / RF : class_weight="balanced"
      XGB     : scale_pos_weight = n_neg / n_pos

    For German the positive class is the majority, so scale_pos_weight is below
    1 (about 0.43). The value is computed from the actual train split rather
    than hardcoded.
    """
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    scale_pos_weight = neg / pos

    lr = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )

    rf = RandomForestClassifier(
        n_estimators=500,
        min_samples_split=20,
        min_samples_leaf=4,
        max_features="log2",
        max_samples=0.7,
        max_depth=15,
        class_weight="balanced",
        n_jobs=1,
        random_state=RANDOM_STATE,
    )

    xgb = XGBClassifier(
        n_estimators=800,
        learning_rate=0.01,
        max_depth=3,
        subsample=0.7,
        colsample_bytree=0.7,
        min_child_weight=3,
        gamma=0,
        reg_lambda=1,
        reg_alpha=0,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=1,
    )

    lr_pipe = Pipeline(steps=[("scaler", StandardScaler()), ("classifier", lr)])
    rf_pipe = Pipeline(steps=[("classifier", rf)])
    xgb_pipe = Pipeline(steps=[("classifier", xgb)])

    return {"LogisticRegression": lr_pipe, "RandomForest": rf_pipe, "XGB": xgb_pipe}


def cross_validate_on_train(
    models: dict[str, Pipeline], X_train: pd.DataFrame, y_train: pd.Series
) -> dict[str, pd.DataFrame]:
    """
    5-fold stratified CV on the train split, identical protocol to Adult
    (same scoring list, StratifiedKFold with n_splits=N_FOLDS / shuffle /
    random_state=42). Returns per-model frames that carry the 5 per-fold rows
    plus a `mean` and a `std` row, so the saved CSV matches the Adult layout.
    """
    scoring_list = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    models_cv_score = {}
    for name, pipe in models.items():
        cv_score = cross_validate(
            pipe, X_train, y_train, cv=cv, scoring=scoring_list, n_jobs=-1
        )
        cv_df = pd.DataFrame(cv_score)
        pick_cols = [i for i in cv_df.columns if i.startswith("test_")]
        cv_df = cv_df[pick_cols]
        fold_scores = cv_df.copy()
        cv_df.loc["mean"] = fold_scores.mean()
        cv_df.loc["std"] = fold_scores.std()
        models_cv_score[name] = cv_df
        print(f"\n[CV] {name}")
        print(cv_df)

    return models_cv_score


def eval_on_test(
    fitted_models: dict[str, Pipeline], X_test: pd.DataFrame, y_test: pd.Series
) -> pd.DataFrame:
    """
    Held-out test evaluation, identical schema to Adult eval_on_test:
    columns `model, accuracy, precision, recall, f1, roc_auc_score`
    (note: test uses `roc_auc_score`, CV uses `test_roc_auc`, kept asymmetric on
    purpose to match the other datasets). roc_auc uses predict_proba[:, 1].
    """
    rows = []
    for name, pipe in fitted_models.items():
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]  # positive col
        rows.append(
            {
                "model": name,
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1": f1_score(y_test, y_pred, zero_division=0),
                "roc_auc_score": roc_auc_score(y_test, y_proba),
            }
        )
    results = pd.DataFrame(rows).set_index("model")
    print("\n[TEST]\n", results)
    return results


def save_baseline_results(
    cv_scores: dict[str, pd.DataFrame],
    test_scores: pd.DataFrame,
    results_dir: Path = RESULTS_DIR,
) -> None:
    """
    Persist baseline CV + test results with the exact structure of the Adult
    files so visualization r1_baseline_performance reads them via a German
    config:

      * german_baseline_cv.csv   -> model, fold, test_accuracy, test_precision,
                                     test_recall, test_f1, test_roc_auc
                                     (fold: 0..4 + mean + std per model)
      * german_baseline_test.csv -> model, accuracy, precision, recall, f1,
                                     roc_auc_score
    """
    results_dir.mkdir(parents=True, exist_ok=True)

    # CV scores: stack all models, insert model + fold columns
    cv_rows = []
    for model_name, cv_df in cv_scores.items():
        temp = cv_df.copy()
        temp.insert(0, "model", model_name)
        temp.insert(1, "fold", temp.index)
        cv_rows.append(temp)

    cv_all = pd.concat(cv_rows, ignore_index=True)
    cv_path = results_dir / "german_baseline_cv.csv"
    cv_all.to_csv(cv_path, index=False, float_format="%.4f")
    print(f"CV scores saved -> {cv_path}")

    # Test scores: model is the index -> keep it as the leading column
    test_path = results_dir / "german_baseline_test.csv"
    test_scores.to_csv(test_path, float_format="%.4f")
    print(f"Test scores saved -> {test_path}")


# =====================================================================
# 3b. Baseline fairness evaluation (mirror adult_fairness)
# =====================================================================
def run_fairness_eval_for_all(
    fitted_models: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, dict[str, tuple[dict, pd.DataFrame]]]:
    """
    Run baseline fairness evaluation for all sensitive attributes and all
    models on the held-out test set, then save. Same logic and shape as
    adult_fairness.run_fairness_eval_for_all (German sensitive attrs
    sex/age_group).
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


def save_fairness_results(
    all_results: dict[str, dict[str, tuple[dict, pd.DataFrame]]],
    results_dir: Path = RESULTS_DIR,
):
    """
    Save baseline fairness evaluation results to disk, schema-identical to the
    NHANES and Adult files:

      1. detail  CSV -> model, sensitive_attr, group, selection_rate,
                        true_positive_rate, false_positive_rate, count,
                        positive_rate
      2. summary CSV -> model, sensitive_attr, Equal Opportunity,
                        Equalized Odds, Demographic Parity
    """
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

    detail_path = results_dir / "german_fairness_baseline_detail.csv"
    summary_path = results_dir / "german_fairness_baseline_summary.csv"

    detail_df.to_csv(detail_path, index=False, float_format="%.4f")
    summary_df.to_csv(summary_path, index=False, float_format="%.4f")

    print(f"Fairness detail  saved -> {detail_path}")
    print(f"Fairness summary saved -> {summary_path}")
    return detail_df, summary_df


def get_fitted_models_and_split_data():
    X, y, raw_df = load_german()

    dataset_profiler(raw_df, target="good_credit", sens_cols=["sex", "age_group"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    models = get3_pipe_models(y_train)
    models_cv_score = cross_validate_on_train(models, X_train, y_train)

    for pipe in models.values():
        pipe.fit(X_train, y_train)

    test_results = eval_on_test(models, X_test, y_test)
    save_baseline_results(models_cv_score, test_results)

    # baseline fairness detail/summary, same models & split as the mitigation grid
    run_fairness_eval_for_all(models, X_test, y_test)

    return models, X_train, X_test, y_train, y_test


# =====================================================================
# 4. Metrics helpers (mirror adult_fairness)
# =====================================================================
def _get_model_performance(y_test, y_pred, y_proba=None) -> dict:
    row = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }
    row["roc_auc"] = roc_auc_score(y_test, y_proba) if y_proba is not None else np.nan
    return row


def _get_model_fairness_eval(y_test, y_pred, sensitive_col):
    return run_fairness_eval(y_true=y_test, y_pred=y_pred, sensitive_col=sensitive_col)


def _build_row(model_name, sensitive_attr, method, performance, fairness_summary):
    return {
        "model": model_name,
        "sensitive_attr": sensitive_attr,
        "miti_method": method,
        **performance,
        **fairness_summary,
    }


def _append_detail(detail_rows, details_df, model_name, attr_name, method):
    for group_name, row in details_df.iterrows():
        detail_rows.append(
            {
                "model": model_name,
                "sensitive_attr": attr_name,
                "miti_method": method,
                "group": group_name,
                **row.to_dict(),
            }
        )


# =====================================================================
# 5. Full mitigation grid
# =====================================================================
def run_all_mitigations() -> list[dict]:
    """
    3 models x 2 sensitive attrs x (Baseline + 4 mitigations) = 30 result rows.
    Uses the same generic mitigation and fairness_eval calls as Adult and NHANES.
    """
    # total wall-clock timer over the ENTIRE mitigation grid (incl. data load,
    # baseline CV/test, and every model x attr x method combination)
    t0 = time.perf_counter()

    Baseline_models, X_train, X_test, y_train, y_test = (
        get_fitted_models_and_split_data()
    )

    sensitive_train_df = extract_sensitive_attrs(X_train)
    sensitive_test_df = extract_sensitive_attrs(X_test)
    suppression_cols = _suppression_cols(X_train)

    all_results: list[dict] = []
    detail_rows: list[dict] = []

    for name, pipe in Baseline_models.items():
        y_pred_base = pipe.predict(X_test)
        y_proba_base = pipe.predict_proba(X_test)[:, 1]
        performance_base = _get_model_performance(y_test, y_pred_base, y_proba_base)

        for col in sensitive_train_df.columns:
            sens_train = sensitive_train_df[col]
            sens_test = sensitive_test_df[col]

            # 1. Baseline
            fairness_summary_base, fairness_details_base = _get_model_fairness_eval(
                y_test, y_pred_base, sens_test
            )
            _append_detail(detail_rows, fairness_details_base, name, col, "Baseline")
            all_results.append(
                _build_row(name, col, "Baseline", performance_base, fairness_summary_base)
            )

            # 2. Reweighing
            print(f" >>>> {name} | {col} | Reweighing ...")
            rw_model = mitigation.fit_model_with_reweighing(
                pipe, X_train, y_train, sens_train
            )
            y_pred_rw = rw_model.predict(X_test)
            y_proba_rw = rw_model.predict_proba(X_test)[:, 1]
            performance_rw = _get_model_performance(y_test, y_pred_rw, y_proba_rw)
            fairness_summary_rw, fairness_details_rw = _get_model_fairness_eval(
                y_test, y_pred_rw, sens_test
            )
            _append_detail(detail_rows, fairness_details_rw, name, col, "Reweighing")
            all_results.append(
                _build_row(name, col, "Reweighing", performance_rw, fairness_summary_rw)
            )

            # 3. ExponentiatedGradient
            print(f" >>>> {name} | {col} | ExponentiatedGradient ...")
            exg_model = mitigation.apply_exponentiated_gradient(
                pipe, X_train, y_train, sens_train
            )
            y_pred_exg = exg_model.predict(X_test, random_state=RANDOM_STATE)
            performance_exg = _get_model_performance(y_test, y_pred_exg)
            fairness_summary_exg, fairness_details_exg = _get_model_fairness_eval(
                y_test, y_pred_exg, sens_test
            )
            _append_detail(
                detail_rows, fairness_details_exg, name, col, "ExponentiatedGradient"
            )
            all_results.append(
                _build_row(
                    name, col, "ExponentiatedGradient", performance_exg, fairness_summary_exg
                )
            )

            # 4. ThresholdOptimizer
            print(f" >>>> {name} | {col} | ThresholdOptimizer ...")
            to_model = mitigation.apply_threshold_optimizer(
                pipe, X_train, y_train, sens_train
            )
            y_pred_to = to_model.predict(
                X_test, random_state=RANDOM_STATE, sensitive_features=sens_test
            )
            performance_to = _get_model_performance(y_test, y_pred_to)
            fairness_summary_to, fairness_details_to = _get_model_fairness_eval(
                y_test, y_pred_to, sens_test
            )
            _append_detail(detail_rows, fairness_details_to, name, col, "ThresholdOptimizer")
            all_results.append(
                _build_row(name, col, "ThresholdOptimizer", performance_to, fairness_summary_to)
            )

            # 5. Suppression
            print(f" >>>> {name} | {col} | Suppression ...")
            sup_model, keep_cols = mitigation.apply_suppression(
                pipe, X_train, y_train, suppression_cols[col]
            )
            X_test_reduced = X_test[keep_cols]
            y_pred_sup = sup_model.predict(X_test_reduced)
            y_proba_sup = sup_model.predict_proba(X_test_reduced)[:, 1]
            performance_sup = _get_model_performance(y_test, y_pred_sup, y_proba_sup)
            fairness_summary_sup, fairness_details_sup = _get_model_fairness_eval(
                y_test, y_pred_sup, sens_test
            )
            _append_detail(detail_rows, fairness_details_sup, name, col, "Suppression")
            all_results.append(
                _build_row(name, col, "Suppression", performance_sup, fairness_summary_sup)
            )

    _save_results(pd.DataFrame(all_results), pd.DataFrame(detail_rows))

    elapsed = time.perf_counter() - t0
    print(f"\nGerman mitigation total time: {elapsed:.1f}s")

    return all_results


def _save_results(results_df: pd.DataFrame, detail_df: pd.DataFrame) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results_path = RESULTS_DIR / "german_mitigation_results.csv"
    detail_path = RESULTS_DIR / "german_mitigation_detail.csv"

    results_df.to_csv(results_path, index=False, float_format="%.4f")
    detail_df.to_csv(detail_path, index=False, float_format="%.4f")

    print(f"\nMitigation results saved -> {results_path}")
    print(results_df.to_string(index=False))
    print(f"\nMitigation detail  saved -> {detail_path}")


def run_profile_only() -> None:
    """Load and profile the dataset only, without training or writing anything."""
    _, _, raw_df = load_german()
    dataset_profiler(raw_df, target="good_credit", sens_cols=["sex", "age_group"])


if __name__ == "__main__":
    run_all_mitigations()
