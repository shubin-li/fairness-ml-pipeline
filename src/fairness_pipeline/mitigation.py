"""
Author:Shubin Li


https://fairlearn.org/
https://ai-fairness-360.org/
https://aif360.readthedocs.io/en/stable/

mitigation selection & theory reference:
1.Reweighing(pre-processing|AIF360):
    Paper: Data preprocessing techniques for classification without discrimination https://link.springer.com/article/10.1007/s10115-011-0463-8
2.ExponentiatedGradient(redutions&in-processing| Fairlearn):
    Paper:A reductions approach to fair classification.  http://proceedings.mlr.press/v80/agarwal18a.html. https://arxiv.org/abs/1803.02453
3.Threshold optimizer(post-processing|Fairlearn):
    Paper:Equality of opportunity in supervised learning.  https://proceedings.neurips.cc/paper/2016/hash/9d2682367c3935defcb1f9e247a97c0d-Abstract.html.
4.Suppression (naive drop sensitive cols| implement by ourself) as baseline contrast:
    Paper:Fairness definitions explained  https://dl.acm.org/doi/abs/10.1145/3194770.3194776


Four mitigation methods spanning pre / in / post processing + baseline contrast:
  1. Reweighing          (pre-processing,  AIF360)  — reweight training samples
  2. ExponentiatedGradient (in-processing, Fairlearn) — train with fairness constraint
  3. ThresholdOptimizer   (post-processing, Fairlearn) — adjust decision thresholds per group
  4. Suppression          (baseline contrast)          — drop sensitive features, retrain
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.base import clone
import sklearn

sklearn.set_config(enable_metadata_routing=True)

from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.preprocessing import Reweighing
from fairlearn.reductions import ExponentiatedGradient, DemographicParity, EqualizedOdds
from fairlearn.postprocessing import ThresholdOptimizer


# reweighing (pre-processing, AIF360) — reweight training samples
# Returns sample_weights that the caller passes to model.fit(sample_weight=)
# due to the sensitive_col being split into two groups, privileged and unprivileged, the sample_weights have 4 unique values
# sensitive_col: pd.Series of sensitive attribute values aligned with X_train!! index
def apply_reweighing(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sensitive_col: pd.Series,
) -> np.ndarray:

    sensitive_attr_name = "sensitive_col"
    # build a temporary df for AIF360
    tmp = X_train.copy()

    # factorize the sensitive_col to numeric values for AIF360
    numeric_sensitive_col, uniques = pd.factorize(sensitive_col)
    tmp[sensitive_attr_name] = numeric_sensitive_col
    tmp["label"] = y_train.values

    # AIF360 needs to know privileged / unprivileged groups
    # we treat every unique value as a group; privileged = majority group
    group_counts = pd.Series(numeric_sensitive_col).value_counts()
    privileged_val = group_counts.idxmax()

    privileged_groups = [{sensitive_attr_name: privileged_val}]
    unprivileged_groups = [
        {sensitive_attr_name: v} for v in group_counts.index if v != privileged_val
    ]

    aif_dataset = BinaryLabelDataset(
        df=tmp,
        label_names=["label"],
        protected_attribute_names=[sensitive_attr_name],
    )

    rw = Reweighing(
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups,
    )
    out = rw.fit_transform(aif_dataset)
    sample_weights = out.instance_weights

    return sample_weights


# If the model is a pipeline, we need to route the sample_weight to the classifier step
def _handle_pipeline_sample_weight(clone_model):

    if isinstance(clone_model, Pipeline):
        for name, step in clone_model.steps[:-1]:
            step.set_fit_request(sample_weight=False)
        clone_model.steps[-1][1].set_fit_request(sample_weight=True)
    return clone_model


# execute reweighing and fit the model with sample weights
def fit_model_with_reweighing(
    model, X_train: pd.DataFrame, y_train: pd.Series, sensitive_col: pd.Series
):
    sample_weights = apply_reweighing(X_train, y_train, sensitive_col)
    clone_model = clone(model)

    clone_model = _handle_pipeline_sample_weight(clone_model)

    clone_model.fit(X_train, y_train, sample_weight=sample_weights)

    return clone_model


# return weighted ensemble model -> ExponentiatedGradient
def apply_exponentiated_gradient(
    model, X_train: pd.DataFrame, y_train: pd.Series, sensitive_col: pd.Series
):

    # Define the fairness constraint  DemographicParity | EqualizedOdds
    constraint = EqualizedOdds()

    clone_model = clone(model)
    # inner ExponentiatedGradient needs pass sample_weight
    clone_model = _handle_pipeline_sample_weight(clone_model)

    # Create the ExponentiatedGradient object , with default eps=0.01 and max_iter=50
    exp_grad = ExponentiatedGradient(
        estimator=clone_model, constraints=constraint, eps=0.01, max_iter=50
    )

    # Fit the model with the fairness constraint
    exp_grad.fit(X_train, y_train, sensitive_features=sensitive_col)

    return exp_grad
