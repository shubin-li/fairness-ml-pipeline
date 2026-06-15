"""
Shubin Li

train : LR , RF , XGBoost

train_test split
build model Pipeline
do stratified cross validate on train
fit final models
evaluate on test
"""

import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_FOLDS = 5


def _train_test_split(df: pd.DataFrame):
    X = df.drop(columns=["depressed"])
    y = df["depressed"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    return X_train, X_test, y_train, y_test


def get3_pipe_models(y_train: pd.Series) -> dict[str, Pipeline]:
    """
    class imbalance  ~13% depressed, handle it with
    LR/RF  : class_weight="balanced"
    XGBoost: scale_pos_weight = n_neg / n_pos
    """

    # pipeline will do Standardization together with train, without data leakage. (when doing CV, it will standardize only on train data)
    # negative  /  positive
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos_weight = neg / pos

    lr = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )

    """
    Fitting 5 folds for each of 50 candidates, totalling 250 fits

    RF Best params: {'n_estimators': 500, 'min_samples_split': 20, 'min_samples_leaf': 4, 'max_features': 'log2', 'max_depth': None}
    RF Best CV roc_auc: 0.7415
    Test AUC: 0.7452 vs Train AUC: 0.9833  vs  CV AUC: 0.7415


    Fitting 5 folds for each of 50 candidates, totalling 250 fits

    RF Best params: {'n_estimators': 500, 'min_samples_split': 20, 'min_samples_leaf': 4, 'max_samples': 0.7, 'max_features': 'log2', 'max_depth': 15}
    RF Best CV roc_auc: 0.7417
    Test AUC: 0.7467 vs Train AUC: 0.9637  vs  CV AUC: 0.7417



    note: learn from error
    RF with n_estimators=300 and all other hyperparameters default
    (min_samples_leaf=1, min_samples_split=2), under 5-fold CV:
    RandomForest
            test_accuracy  test_precision  test_recall   test_f1  test_roc_auc
    0          0.868270        0.000000     0.000000  0.000000      0.710766
    1          0.865979        0.000000     0.000000  0.000000      0.730601
    2          0.868270        0.666667     0.017241  0.033613      0.699506
    3          0.865979        0.000000     0.000000  0.000000      0.736408
    4          0.865826        0.000000     0.000000  0.000000      0.739969
    mean       0.866865        0.133333     0.003448  0.006723      0.723450
    std        0.001149        0.266667     0.006897  0.013445      0.015660

    Accuracy looks fine and AUC is normal, yet recall is ~0. The model
    predicts almost every sample as the majority class (No Depression).
    This is easy to miss: tuning with scoring='roc_auc' alone never exposes
    it, because AUC is threshold-independent.

    UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))


    """

    # rf = RandomForestClassifier(
    #     n_estimators=500,
    #     min_samples_split=20,
    #     min_samples_leaf=4,
    #     max_features="log2",
    #     class_weight="balanced",
    #     n_jobs=-1,
    #     random_state=RANDOM_STATE,
    # )
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
    """
    Fitting 5 folds for each of 50 candidates, totalling 250 fits

    XGB Best params: {'subsample': 0.7, 'reg_lambda': 1, 'reg_alpha': 0, 'n_estimators': 800, 'min_child_weight': 3, 'max_depth': 3, 'learning_rate': 0.01, 'gamma': 0, 'colsample_bytree': 0.7}
    XGB Best CV roc_auc: 0.7397
    Test AUC: 0.7467 vs Train AUC: 0.8466  vs  CV AUC: 0.7397
    """
    # xgb = XGBClassifier(
    #     n_estimators=300,
    #     learning_rate=0.1,
    #     max_depth=4,
    #     subsample=0.9,
    #     colsample_bytree=0.9,
    #     scale_pos_weight=scale_pos_weight,
    #     eval_metric="logloss",
    #     random_state=RANDOM_STATE,
    #     n_jobs=-1,
    # )

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

    # put StandardScaler in pipeline, when doing cross_validation, we can keep it only fit with train rather than validation part
    lr_pipe = Pipeline(steps=[("scaler", StandardScaler()), ("classifier", lr)])
    rf_pipe = Pipeline(steps=[("classifier", rf)])
    xgb_pipe = Pipeline(steps=[("classifier", xgb)])

    models = {"LogisticRegression": lr_pipe, "RandomForest": rf_pipe, "XGB": xgb_pipe}

    return models


# 5-fold stratifiedCV on train data
def cross_validate_on_train(
    models: dict[str, Pipeline], X_train: pd.DataFrame, y_train: pd.Series
) -> dict[str, pd.DataFrame]:
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
        fold_scores=cv_df.copy()
        cv_df.loc["mean"] = fold_scores.mean()
        cv_df.loc["std"] = fold_scores.std()
        models_cv_score[name] = cv_df
        print(f"\n{name}")
        print(cv_df)

    return models_cv_score


def fit_final_models(
    models: dict[str, Pipeline], X_train: pd.DataFrame, y_train: pd.Series
) -> dict[str, Pipeline]:
    for pipe in models.values():
        pipe.fit(X_train, y_train)

    return models


def eval_on_test(
    fitted_models: dict[str, Pipeline], X_test: pd.DataFrame, y_test: pd.Series
) -> pd.DataFrame:

    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
    )

    rows = []
    for name, pipe in fitted_models.items():
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]  # positive col
        row = {
            "model": name,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc_score": roc_auc_score(y_test, y_proba),
        }
        rows.append(row)
    results = pd.DataFrame(rows).set_index("model")
    print("\n", results)
    return results


def run_all_models() -> dict[str, Pipeline]:
    import preprocessing

    df = preprocessing.run_all_preprocessing()
    X_train, X_test, y_train, y_test = _train_test_split(df)
    models = get3_pipe_models(y_train)
    models_cv_score = cross_validate_on_train(models, X_train, y_train)
    fitted_models = fit_final_models(models, X_train, y_train)
    results = eval_on_test(fitted_models, X_test, y_test)
    return fitted_models


if __name__ == "__main__":
    run_all_models()
