# Fairness Audit Pipeline for High-Risk ML Applications

A model-agnostic pipeline covering bias **detection, mitigation, and reassessment**, validated under one frozen experimental protocol on six datasets spanning healthcare, education, finance, and income prediction (positive prevalence 0.12 to 0.70). MSc Practicum, Dublin City University, 2026.

> ⚠️ **GitHub may fail to render large notebooks.** View the extension notebooks on [nbviewer](https://nbviewer.org/github/shubin-li/ml-fairness-audit-pipeline/tree/master/src/partner-extensions/)

## Overview

The EU AI Act classifies medical prediction and student performance evaluation as high-risk applications and requires fairness assessment before deployment. Existing fairness work mostly evaluates a single prediction task, studies detection and mitigation in isolation, and rarely validates a reusable method across datasets.

This project closes that gap with a pipeline whose evaluation and mitigation modules contain no dataset-specific logic. A new dataset connects through one lightweight adapter and inherits the full experimental matrix: 3 classifiers × (Baseline + 4 treatments) × all sensitive attributes. The Baseline arm serves as detection, the mitigated arms as reassessment.

Two lightweight diagnostic criteria run as pure post-processing on the evaluation output, adding zero extra experiments:

| Criterion | What it catches |
|---|---|
| **False-Fairness Criterion** | A degenerate model reading as perfectly fair. Flags `min TPR < 0.05` combined with `ΔEOpp < 0.05` |
| **Model-Manufactured Disparity Metric** | Splits an observed selection-rate gap into the data-inherent share and the model-added share |

### Key Results

| Finding | Evidence |
|---|---|
| Standard metrics can certify a collapsed model as fair | NHANES / LR / income / ExponentiatedGradient: 1 positive prediction across 1,091 test instances, recall 0.000, yet ΔEOpp = 0.000 and ΔDP = 0.001. Flagged as the single degenerate case among 105 configurations |
| Models amplify disparity beyond the data | True base-rate gap 0.133 becomes an output selection-rate gap of 0.588, a model-added share of 0.456 |
| Dropping the sensitive column does not remove bias | Suppression retains 0.212 (NHANES income) and 0.218 (NHANES race) of model-added disparity through proxy features |
| Mitigation conclusions flip with the base classifier | NHANES race under EG: LR 0.640 → 0.161 (−75%), XGBoost 0.677 → 0.628 (−7%). OULAD region under ThresholdOptimizer: RF worsens from 0.140 to 0.488 |
| Binary attributes behave consistently | Gender gaps reduce to at or near 0.05 on every dataset at negligible F1 cost |

## Methodology

### Dataset Characterisation

Datasets are organised by four measurable properties rather than by application domain: positive prevalence, sensitive-attribute cardinality, subgroup base-rate gap, and subgroup sample size.

| Dataset | Domain | n | Prevalence | Sensitive attributes (cardinality) |
|---|---|---|---|---|
| NHANES 2021–2023 | Healthcare | 5,455 | 0.13 | gender (2), income (3), race (5) |
| Adult Income | Income | 48,842 | 0.24 | sex (2), race (5) |
| German Credit | Credit | 1,000 | 0.70 | sex (2), age group (2) |
| OULAD | Education | 32,593 | 0.47 | gender (2), region (13), age band (2) |
| Bank Marketing | Finance | 45,211 | 0.12 | age, marital status, education (2 each) |
| Credit Card Clients | Finance | 30,000 | 0.22 | gender (2) |

### Treatments

One method per lifecycle stage plus one control, each applicable to an arbitrary base classifier:

- **Reweighing** (pre-processing, AIF360): sample weights from the joint distribution of sensitive attribute and label
- **ExponentiatedGradient** (in-processing, Fairlearn): reductions under an equalized odds constraint, slack 0.01, 50 iterations
- **ThresholdOptimizer** (post-processing, Fairlearn): per-subgroup decision thresholds, 20% internal validation split to avoid leakage
- **Suppression** (control): retrain after removing the sensitive column, testing the assumption that blindness produces fairness

### Evaluation Protocol

Identical for all datasets: stratified 80/20 split, 5-fold stratified cross-validation on the training set, final fit on the full training set, all results reported on the held-out test set, seed 42. Classifiers are Logistic Regression, Random Forest, and XGBoost, with class imbalance handled uniformly through balanced class weights and scale_pos_weight. Group-level output records selection rate, TPR, FPR, subgroup size, and true prevalence; aggregate output reports Demographic Parity, Equal Opportunity, and Equalized Odds differences as max minus min across subgroups.

## Tech Stack

- **Language:** Python 3.11
- **Fairness:** Fairlearn (MetricFrame, ExponentiatedGradient, ThresholdOptimizer), AIF360 (Reweighing)
- **ML:** scikit-learn, XGBoost
- **Data Processing:** pandas, NumPy, pyreadstat (NHANES XPT files)
- **Visualisation:** matplotlib, seaborn

## Project Structure

```
src/
├── fairness_pipeline/          # Frozen dataset-agnostic core
│   ├── fairness_eval.py        # Group-level and aggregate metric computation
│   ├── mitigation.py           # Four treatments behind a uniform interface
│   └── diagnostics.py          # False-Fairness Criterion + Model-Manufactured Disparity
├── nhanes/                     # Healthcare adapter (low-prevalence anchor)
│   ├── data_loader.py          # NHANES XPT ingestion and module merge
│   ├── preprocessing.py        # PHQ-9 label construction, sensitive-attribute extraction
│   ├── models.py               # Classifier configuration and tuning
│   ├── nhanes_fairness.py      # Experiment driver
│   ├── visualization.py
│   └── notebooks/EDA.ipynb
├── benchmark/                  # Adult Income and German Credit adapters
├── partner-extensions/         # OULAD, Bank Marketing, Credit Card Clients
├── results/                    # Per-dataset metric exports and figures
└── data/nhanes/                # Raw NHANES 2021–2023 XPT files
```

## My Contributions

Two-person practicum. My responsibility was the pipeline core and the three anchor datasets:

- **Pipeline architecture:** the dataset-agnostic evaluation and mitigation modules, the adapter interface that keeps a new dataset to one loader, and the uniform experimental matrix that makes results comparable across datasets
- **Diagnostic layer:** design and implementation of both criteria, including the threshold sensitivity check (τ = 0.03, 0.05, 0.10 identify the same single flag) and the documented blind spot toward the symmetric all-positive collapse
- **NHANES adapter:** module merge across ten survey files, PHQ-9 depression label construction, income-to-poverty ratio binning, and the full experiment run across three sensitive attributes
- **Benchmark adapters:** Adult Income and German Credit, selected as the mid- and high-prevalence anchors to span the prevalence coordinate
- **Analysis:** the three research questions, the cross-dataset gender comparison, and the mechanism discussion covering constrained-optimiser collapse, proxy leakage under Suppression, and multi-category gap structure

## Data Sources

| Dataset | Source |
|---|---|
| NHANES 2021–2023 | [CDC / NCHS](https://wwwn.cdc.gov/nchs/nhanes/) |
| Adult Income | [UCI ML Repository](https://doi.org/10.24432/C5XW20) |
| German Credit | [UCI ML Repository](https://doi.org/10.24432/C5NC77) |
| OULAD | [Open University Learning Analytics Dataset](https://analyse.kmi.open.ac.uk/open-dataset) |
| Bank Marketing | UCI ML Repository (Moro, Cortez & Rita, 2014) |
| Credit Card Clients | UCI ML Repository (Yeh & Lien, 2009) |

## Authors

- **Shubin Li** — pipeline core, diagnostic criteria, NHANES and benchmark experiments
- **Lamaan Kayum Shaikh** — extension datasets and supporting analysis

Supervisor: Dr. Tai Tan Mai. MSc in Computing (Data Analytics), Dublin City University, 2026.
