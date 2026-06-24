"""
Author: Shubin Li

generate NHANES figures for both in paper and viva

"""

import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import seaborn as sns
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results" / "nhanes"
FIG = RESULTS_DIR / "figures"

# Data
mit_df = pd.read_csv(RESULTS_DIR / "nhanes_mitigation_results.csv")
cv_df = pd.read_csv(RESULTS_DIR / "nhanes_baseline_cv.csv")
test_df = pd.read_csv(RESULTS_DIR / "nhanes_baseline_test.csv")
detail_df = pd.read_csv(RESULTS_DIR / "nhanes_fairness_baseline_detail.csv")
summary_df = pd.read_csv(RESULTS_DIR / "nhanes_fairness_baseline_summary.csv")

# config

# cividis viridis magma
HEATMAP_CMAP = "cividis"

METHOD_COLORS = {
    "Baseline": "#2c3e50",
    "Reweighing": "#2980b9",
    "ExponentiatedGradient": "#e67e22",
    "ThresholdOptimizer": "#27ae60",
    "Suppression": "#8e44ad",
}
METHOD_ORDER = [
    "Baseline",
    "Reweighing",
    "ExponentiatedGradient",
    "ThresholdOptimizer",
    "Suppression",
]
METHOD_SHORT = {
    "Baseline": "Base",
    "Reweighing": "RW",
    "ExponentiatedGradient": "EG",
    "ThresholdOptimizer": "TO",
    "Suppression": "Supp",
}
MODEL_ORDER = ["LogisticRegression", "RandomForest", "XGB"]
MODEL_SHORT = {"LogisticRegression": "LR", "RandomForest": "RF", "XGB": "XGB"}
MODEL_COLORS = ["#2c3e50", "#2980b9", "#e67e22"]
ATTR = ["gender", "income", "race"]
ATTR_TITLES = {"gender": "Gender", "income": "Income Group", "race": "Race Group"}
FAIR_METRICS = ["Demographic Parity", "Equal Opportunity", "Equalized Odds"]

# group display order
GROUP_ORDER = {
    "gender": ["Female", "Male"],
    "income": ["Above threshold", "Near poverty", "Low income"],
    "race": ["Asian", "Black", "Hispanic", "Other Race", "White"],
}


def ieee_style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.5,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
        }
    )


def ppt_style():
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 16,
            "axes.titlesize": 20,
            "axes.labelsize": 17,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "legend.fontsize": 14,
            "figure.dpi": 150,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.1,
        }
    )


def save(fig, name):
    p = FIG / name
    # fig.savefig(f"{p}.pdf");
    fig.savefig(f"{p}.png")
    plt.close(fig)
    print(f">>> saving: {name}")


# get sample size of a group, from detail_df
def group_n(attr, group):
    r = detail_df[(detail_df["sensitive_attr"] == attr) & (detail_df["group"] == group)]
    return int(r["count"].iloc[0]) if len(r) else 0



"""
figure 1 : Baseline Model Performance (CV vs Test)
Finding:
the (a) graph shows :
1.low std -> stable not fold dependent
2.RF highest accuray,but lowest recall 
because class imbalance: at 13% positive rate + 0.5 threshold. RF predict almost all negative
3. AUC is fine, because it not dependent the threshold
4. XGB: most balanced (AUC 0.75 + recall 0.62) → best baseline
5. All F1 scores low (0.32-0.37), because data imbalance

the (b) graph shows:
there is no overfitting (CV≈Test)
 
why CV vs test not train vs test?
Train vs Test is uninformative for RF/XGB. 
RF train AUC=0.9637, XGB train AUC= 0.8466.
This is the ensemble model's structural feature. They memorize
training data.

CV evaluates on unseen validation folds within the training set,
same principle as the held-out test set. So CV vs Test can verify generalization.
"""
def r1_baseline_performance():
    ieee_style()
    cvm = cv_df[cv_df['fold']=='mean']; cvs = cv_df[cv_df['fold']=='std']
    cv_cols  = ['test_accuracy','test_precision','test_recall','test_f1','test_roc_auc']
    test_cols= ['accuracy','precision','recall','f1','roc_auc_score']
    labels   = ['Accuracy','Precision','Recall','F1','AUC-ROC']

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.0))
    x = np.arange(len(labels)); w = 0.25

    # (a) CV mean ± std
    ax = axes[0]
    for i, model in enumerate(MODEL_ORDER):
        m = cvm[cvm['model']==model][cv_cols].values.flatten().astype(float)
        s = cvs[cvs['model']==model][cv_cols].values.flatten().astype(float)
        bars = ax.bar(x+i*w, m, w, yerr=s, label=MODEL_SHORT[model], color=MODEL_COLORS[i],
                      capsize=2, error_kw={'linewidth':0.8}, edgecolor='white', linewidth=0.5)
        # value labels above the error bars
        for bar, mv, sv in zip(bars, m, s):
            ax.text(bar.get_x()+bar.get_width()/2, mv+sv+0.015,
                    f'{mv:.2f}', ha='center', va='bottom', fontsize=6)
    ax.set_xticks(x+w); ax.set_xticklabels(labels); ax.set_ylim(0,1.05)
    ax.set_ylabel('Score'); ax.set_title('(a) 5-Fold Stratified CV (Mean ± Std)')
    ax.spines[['top','right']].set_visible(False)
    # legend ABOVE the axis -> never covers bars
    ax.legend(loc='lower center', bbox_to_anchor=(0.5,1.10), ncol=3, frameon=False)

    # (b) CV vs Test — ALL 5 metrics
    ax2 = axes[1]
    x2 = np.arange(len(labels)); w2 = 0.13
    for i, model in enumerate(MODEL_ORDER):
        cvals = cvm[cvm['model']==model][cv_cols].values.flatten().astype(float)
        tvals = test_df[test_df['model']==model][test_cols].values.flatten().astype(float)
        ax2.bar(x2+i*2*w2,    cvals, w2, color=MODEL_COLORS[i], alpha=0.55, edgecolor='white', linewidth=0.4)
        ax2.bar(x2+i*2*w2+w2, tvals, w2, color=MODEL_COLORS[i], alpha=1.0,  edgecolor='white', linewidth=0.4)
        for j, tv in enumerate(tvals):
            ax2.text(x2[j]+i*2*w2+w2+w2/2, tv+0.015,
                     f'{tv:.2f}', ha='center', va='bottom', fontsize=5.5)
    ax2.set_xticks(x2+2.5*w2); ax2.set_xticklabels(labels)
    ax2.set_ylim(0,1.05); ax2.set_title('(b) CV vs Test (Generalization)')
    ax2.spines[['top','right']].set_visible(False)
    leg = [Patch(facecolor='gray', alpha=0.55, label='CV'),
           Patch(facecolor='gray', alpha=1.0,  label='Test')]
    ax2.legend(handles=leg, loc='lower center', bbox_to_anchor=(0.5,1.10), ncol=2, frameon=False)

    fig.tight_layout()
    save(fig, "r1_baseline_performance")




















if __name__ == "__main__":
    r1_baseline_performance()
