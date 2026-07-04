"""
Author: Shubin Li

Generate the German Credit figures by reusing the ten parametrized figure
functions from nhanes.visualization. Nothing dataset-specific is duplicated
here, only a GERMAN_CONFIG (VizConfig) is defined and passed in.

"""

from pathlib import Path

import pandas as pd

from nhanes.visualization import (
    VizConfig,
    r1_baseline_performance,
    f1_baseline_details_dumbbell,
    r2_baseline_detais_tpr_fpr,
    r3_fairness_mit_grid_full,
    r4_performance_mit_grid_full,
    r5_disparity_reduction,
    r6_disparity_reduction,
    f2_tradeoff_f1_EOpp,
    f3_method_gap_comparison,
    f4_mit_detail_gap_dumbbell,
    d1_false_fairness_quadrant,
    d2_amplification_decomposition,
)

RESULTS_DIR = Path(__file__).parent.parent / "results" / "benchmark" / "german"
FIG = RESULTS_DIR / "figures"


def build_german_config() -> VizConfig:
    return VizConfig(
        RESULTS_DIR=RESULTS_DIR,
        FIG=FIG,
        mit_df=pd.read_csv(RESULTS_DIR / "german_mitigation_results.csv"),
        cv_df=pd.read_csv(RESULTS_DIR / "german_baseline_cv.csv"),
        test_df=pd.read_csv(RESULTS_DIR / "german_baseline_test.csv"),
        detail_df=pd.read_csv(RESULTS_DIR / "german_fairness_baseline_detail.csv"),
        mit_detail_df=pd.read_csv(RESULTS_DIR / "german_mitigation_detail.csv"),
        # two sensitive attributes: sex and age_group
        ATTR=["sex", "age_group"],
        ATTR_TITLES={"sex": "Sex", "age_group": "Age Group"},
        ATTR_TITLES_SHORT={"sex": "Sex", "age_group": "Age Group"},
        # real group names read from german_fairness_baseline_detail.csv
        GROUP_ORDER={
            "sex": ["Female", "Male"],
            "age_group": ["young", "old"],
        },
    )


GERMAN_CONFIG = build_german_config()


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    # ten parametrized figures, driven by the German config
    r1_baseline_performance(GERMAN_CONFIG)
    f1_baseline_details_dumbbell(GERMAN_CONFIG)
    r2_baseline_detais_tpr_fpr(GERMAN_CONFIG)
    r3_fairness_mit_grid_full(GERMAN_CONFIG)
    r4_performance_mit_grid_full(GERMAN_CONFIG)
    r5_disparity_reduction(GERMAN_CONFIG)
    r6_disparity_reduction(GERMAN_CONFIG)
    f2_tradeoff_f1_EOpp(GERMAN_CONFIG)
    f3_method_gap_comparison(GERMAN_CONFIG)
    f4_mit_detail_gap_dumbbell(GERMAN_CONFIG)
    # diagnostic figures d1 and d2 for German (into german/figures)
    d1_false_fairness_quadrant(GERMAN_CONFIG)
    d2_amplification_decomposition(GERMAN_CONFIG)
