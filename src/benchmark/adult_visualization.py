"""
Author: Shubin Li

Generate the Adult Income figures by reusing the ten parametrized figure
functions from nhanes.visualization. Nothing dataset-specific is duplicated
here — only an ADULT_CONFIG (VizConfig) is defined and passed in.

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
    paper_fig3_adult_method_gaps,
    ppt_fig3_adult_method_gaps
)

RESULTS_DIR = Path(__file__).parent.parent / "results" / "benchmark" / "adult"
FIG = RESULTS_DIR / "figures"


def build_adult_config() -> VizConfig:
    return VizConfig(
        RESULTS_DIR=RESULTS_DIR,
        FIG=FIG,
        mit_df=pd.read_csv(RESULTS_DIR / "adult_mitigation_results.csv"),
        cv_df=pd.read_csv(RESULTS_DIR / "adult_baseline_cv.csv"),
        test_df=pd.read_csv(RESULTS_DIR / "adult_baseline_test.csv"),
        detail_df=pd.read_csv(RESULTS_DIR / "adult_fairness_baseline_detail.csv"),
        mit_detail_df=pd.read_csv(RESULTS_DIR / "adult_mitigation_detail.csv"),
        # two sensitive attributes: sex (cross-domain comparable) and race
        ATTR=["sex", "race"],
        ATTR_TITLES={"sex": "Sex", "race": "Race"},
        ATTR_TITLES_SHORT={"sex": "Sex", "race": "Race"},
        # real group names read from adult_fairness_baseline_detail.csv
        GROUP_ORDER={
            "sex": ["Female", "Male"],
            "race": [
                "Amer-Indian-Eskimo",
                "Asian-Pac-Islander",
                "Black",
                "Other",
                "White",
            ],
        },
    )


ADULT_CONFIG = build_adult_config()


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    # ten parametrized figures — driven by the Adult config
    r1_baseline_performance(ADULT_CONFIG)
    f1_baseline_details_dumbbell(ADULT_CONFIG)
    r2_baseline_detais_tpr_fpr(ADULT_CONFIG)
    r3_fairness_mit_grid_full(ADULT_CONFIG)
    r4_performance_mit_grid_full(ADULT_CONFIG)
    r5_disparity_reduction(ADULT_CONFIG)
    r6_disparity_reduction(ADULT_CONFIG)
    f2_tradeoff_f1_EOpp(ADULT_CONFIG)
    f3_method_gap_comparison(ADULT_CONFIG)
    f4_mit_detail_gap_dumbbell(ADULT_CONFIG)
    # diagnostic figures d1 and d2 for Adult (into adult/figures)
    d1_false_fairness_quadrant(ADULT_CONFIG)
    d2_amplification_decomposition(ADULT_CONFIG)

    paper_fig3_adult_method_gaps(ADULT_CONFIG)
    ppt_fig3_adult_method_gaps(ADULT_CONFIG)

