"""
Shubin Li

public fairness evaluation as interface for both nhanes and oulad

fairness metrics :
Equalized Odds   (max of TPR gap and FPR gap)
Equal Opportunity (TPR gap)
Demographic Parity (selection_rate gap)


"""

import pandas as pd
from fairlearn.metrics import (
    MetricFrame,  # helper class to compute metrics for different groups
    selection_rate,
    true_positive_rate,
    false_positive_rate,
    demographic_parity_difference,
    equalized_odds_difference,
)


# build a function to compute Equal Opportunity Difference (TPR gap) for a given sensitive attribute
def _equal_opportunity_difference(y_true, y_pred, sensitive_features: pd.Series):
    """
    Equal Opportunity Difference (TPR gap) , use max-min of TPR across groups as the gap
    """
    metric_frame = MetricFrame(
        metrics=true_positive_rate,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features,
    )

    return metric_frame.difference(method="between_groups")


# public fairness evaluation interface for both nhanes and oulad
def run_fairness_eval(y_true: pd.Series, y_pred: pd.Series, sensitive_col: pd.Series):

    metric_frame = MetricFrame(
        metrics={
            "selection_rate": selection_rate,
            "true_positive_rate": true_positive_rate,
            "false_positive_rate": false_positive_rate,
        },
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_col,
    )

    details_df = metric_frame.by_group.copy()
    details_df["count"] = sensitive_col.value_counts()
    details_df["positive_rate"] = y_true.groupby(sensitive_col).mean()

    summary = {}

    equal_opportunity_diff = _equal_opportunity_difference(
        y_true, y_pred, sensitive_features=sensitive_col
    )
    equalized_odds_diff = equalized_odds_difference(
        y_true, y_pred, sensitive_features=sensitive_col
    )
    demographic_parity_diff = demographic_parity_difference(
        y_true, y_pred, sensitive_features=sensitive_col
    )

    summary["Equal Opportunity"] = equal_opportunity_diff
    summary["Equalized Odds"] = equalized_odds_diff
    summary["Demographic Parity"] = demographic_parity_diff

    return summary, details_df




