"""
Author:Shubin Li

Post-hoc diagnostics over existing mitigation CSVs.

This module reads mitigation result and detail CSVs that were already
produced by the pipeline and derives two diagnostic tables. It is pure
post-processing.

Tool 1 (False-Fairness Criterion) separates genuine fairness from a
degenerate model whose low disparity hides a near-dead subgroup.

Tool 2 (Model-Manufactured Disparity) splits observed disparity into the
part carried by the data and the part the model adds on top.
"""

import os

import pandas as pd

# Grouping key shared by both tools.
GROUP_KEYS = ["model", "sensitive_attr", "miti_method"]

# Thresholds for the false-fairness rule.
RECALL_FLOOR = 0.05
GAP_FLOOR = 0.05

DATASET_CANDIDATES = ["nhanes", "adult", "german"]


def _dataset_dir(dataset, results_dir):
    """Return the directory holding a dataset's CSVs, or None if absent.

    adult and german live under benchmark/{dataset}. nhanes lives directly
    under results/{dataset}. Detection checks for the results CSV so no
    dataset is assumed present.
    """
    layouts = [
        os.path.join(results_dir, "benchmark", dataset),
        os.path.join(results_dir, dataset),
    ]
    for path in layouts:
        results_csv = os.path.join(path, "{}_mitigation_results.csv".format(dataset))
        detail_csv = os.path.join(path, "{}_mitigation_detail.csv".format(dataset))
        if os.path.isfile(results_csv) and os.path.isfile(detail_csv):
            return path
    return None


def false_fairness(results_df, detail_df):
    """Flag groups where low disparity coexists with a near-dead subgroup.

    Per (model, sensitive_attr, miti_method) the minimum subgroup recall
    comes from true_positive_rate in the detail rows. The gap comes from the
    aggregate Equal Opportunity column in the results table, which is the TPR
    gap. A flag means the disparity looks small only because the model barely
    predicts positives for some subgroup.
    """
    records = []
    for keys, block in detail_df.groupby(GROUP_KEYS, sort=False):
        model, sensitive_attr, miti_method = keys
        min_idx = block["true_positive_rate"].idxmin()
        min_row = block.loc[min_idx]
        min_group_recall = float(min_row["true_positive_rate"])
        group = min_row["group"]
        count = min_row["count"]

        # gap comes from the aggregate Equal Opportunity column (TPR gap).
        match = results_df[
            (results_df["model"] == model)
            & (results_df["sensitive_attr"] == sensitive_attr)
            & (results_df["miti_method"] == miti_method)
        ]
        if match.empty:
            gap = float("nan")
        else:
            gap = float(match.iloc[0]["Equal Opportunity"])

        flag = bool(
            (min_group_recall < RECALL_FLOOR)
            and pd.notna(gap)
            and (gap < GAP_FLOOR)
        )

        records.append(
            {
                "model": model,
                "sensitive_attr": sensitive_attr,
                "miti_method": miti_method,
                "group": group,
                "min_group_recall": min_group_recall,
                "gap": gap,
                "count": count,
                "false_fairness_flag": flag,
            }
        )

    return pd.DataFrame.from_records(records)


def manufactured_disparity(detail_df):
    """Split selection-rate disparity into data-inherent and model-created.

    observed_gap is the spread in selection_rate across subgroups.
    true_prevalence_gap is the spread in positive_rate, the disparity the
    data already carries. amplification is what the model adds. A positive
    amplification means the model manufactures disparity beyond the data.
    """
    records = []
    for keys, block in detail_df.groupby(GROUP_KEYS, sort=False):
        model, sensitive_attr, miti_method = keys

        max_sr_idx = block["selection_rate"].idxmax()
        min_sr_idx = block["selection_rate"].idxmin()
        observed_gap = float(
            block.loc[max_sr_idx, "selection_rate"]
            - block.loc[min_sr_idx, "selection_rate"]
        )
        true_prevalence_gap = float(
            block["positive_rate"].max() - block["positive_rate"].min()
        )
        amplification = observed_gap - true_prevalence_gap

        records.append(
            {
                "model": model,
                "sensitive_attr": sensitive_attr,
                "miti_method": miti_method,
                "observed_gap": observed_gap,
                "true_prevalence_gap": true_prevalence_gap,
                "amplification": amplification,
                "max_sr_group": block.loc[max_sr_idx, "group"],
                "min_sr_group": block.loc[min_sr_idx, "group"],
            }
        )

    return pd.DataFrame.from_records(records)


def run_diagnostics(dataset, results_dir):
    """Read a dataset's two CSVs and write the two diagnostic tables.

    Returns a summary dict with counts.
    Output CSVs land beside the inputs. 
    """
    data_dir = _dataset_dir(dataset, results_dir)
    if data_dir is None:
        return None

    results_path = os.path.join(data_dir, "{}_mitigation_results.csv".format(dataset))
    detail_path = os.path.join(data_dir, "{}_mitigation_detail.csv".format(dataset))

    # roc_auc may be blank for ExponentiatedGradient and ThresholdOptimizer.
    # These tools do not read AUC, so pandas parsing the blank as NaN is fine.
    results_df = pd.read_csv(results_path)
    detail_df = pd.read_csv(detail_path)

    ff_df = false_fairness(results_df, detail_df)
    md_df = manufactured_disparity(detail_df)

    ff_path = os.path.join(data_dir, "{}_false_fairness.csv".format(dataset))
    md_path = os.path.join(data_dir, "{}_manufactured_disparity.csv".format(dataset))


    ff_df.round(4).to_csv(ff_path, index=False)
    md_df.round(4).to_csv(md_path, index=False)

    return {
        "dataset": dataset,
        "false_fairness_flags": int(ff_df["false_fairness_flag"].sum()),
        "positive_amplification": int((md_df["amplification"] > 0).sum()),
        "false_fairness_path": ff_path,
        "manufactured_disparity_path": md_path,
    }


def run_diagnostics_from_paths(results_path, detail_path, output_prefix=None):
    """Read two CSVs from arbitrary paths and write the two diagnostic tables.

    Unlike run_diagnostics this does not assume the DATASET_CANDIDATES naming
    or the benchmark/{dataset} layout. The two paths are read directly and each
    already includes its filename. Output CSVs land in the same directory as
    results_path. When output_prefix is given it names the outputs; otherwise
    the prefix comes from the results filename by stripping a trailing
    _mitigation_results.csv, falling back to the filename stem. Returns a
    summary dict with the same shape as run_diagnostics.
    """
    if not os.path.isfile(results_path):
        raise FileNotFoundError("results CSV not found: {}".format(results_path))
    if not os.path.isfile(detail_path):
        raise FileNotFoundError("detail CSV not found: {}".format(detail_path))

    out_dir = os.path.dirname(results_path)
    if output_prefix is None:
        results_name = os.path.basename(results_path)
        suffix = "_mitigation_results.csv"
        if results_name.endswith(suffix):
            output_prefix = results_name[: -len(suffix)]
        else:
            output_prefix = os.path.splitext(results_name)[0]

    # roc_auc may be blank for ExponentiatedGradient and ThresholdOptimizer.
    # These tools do not read AUC, so pandas parsing the blank as NaN is fine.
    results_df = pd.read_csv(results_path)
    detail_df = pd.read_csv(detail_path)

    ff_df = false_fairness(results_df, detail_df)
    md_df = manufactured_disparity(detail_df)

    ff_path = os.path.join(out_dir, "{}_false_fairness.csv".format(output_prefix))
    md_path = os.path.join(out_dir, "{}_manufactured_disparity.csv".format(output_prefix))

    ff_df.round(4).to_csv(ff_path, index=False)
    md_df.round(4).to_csv(md_path, index=False)

    return {
        "dataset": output_prefix,
        "false_fairness_flags": int(ff_df["false_fairness_flag"].sum()),
        "positive_amplification": int((md_df["amplification"] > 0).sum()),
        "false_fairness_path": ff_path,
        "manufactured_disparity_path": md_path,
    }


def _find_results_dir():
    """Return the path to the results directory, relative to this file."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(os.path.dirname(here), "results")


if __name__ == "__main__":
    results_dir = _find_results_dir()

    found_any = False
    for dataset in DATASET_CANDIDATES:
        summary = run_diagnostics(dataset, results_dir)
        if summary is None:
            continue
        found_any = True
        print(
            "{}: {} false-fairness flags, {} rows with positive amplification".format(
                summary["dataset"],
                summary["false_fairness_flags"],
                summary["positive_amplification"],
            )
        )

    if not found_any:
        print("No mitigation CSVs found under {}".format(results_dir))

    # summary = run_diagnostics_from_paths(
    #     r"src\results\oulad\oulad_mitigation_results__1_.xls",
    #     r"src\results\oulad\oulad_mitigation_detail__1_.csv",
    # )
    # print(
    #     "{}: {} false-fairness flags, {} rows with positive amplification".format(
    #         summary["dataset"],
    #         summary["false_fairness_flags"],
    #         summary["positive_amplification"],
    #     )
    # )

    summary = run_diagnostics_from_paths(
        r"src\results\More Datasets\Credit card\credit_mitigation_results.xls",
        r"src\results\More Datasets\Credit card\credit_mitigation_detail.xls",
        output_prefix="credit_card",
    )
    print(
        "{}: {} false-fairness flags, {} rows with positive amplification".format(
            summary["dataset"],
            summary["false_fairness_flags"],
            summary["positive_amplification"],
        )
    )

    summary = run_diagnostics_from_paths(
        r"src\results\More Datasets\Bank marketing\bank_marketing_mitigation_results_final.csv",
        r"src\results\More Datasets\Bank marketing\bank_marketing_mitigation_detail_final.csv",
        output_prefix="bank_marketing",
    )
    print(
        "{}: {} false-fairness flags, {} rows with positive amplification".format(
            summary["dataset"],
            summary["false_fairness_flags"],
            summary["positive_amplification"],
        )
    )
