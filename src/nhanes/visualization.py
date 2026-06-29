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
mit_detail_df = pd.read_csv(RESULTS_DIR / "nhanes_mitigation_detail.csv")

#oulad data
oulad_df = pd.read_csv(RESULTS_DIR / ".." / "oulad" / "oulad_mitigation_results__1_.xls")

# nhanes config

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

# OULAD config
OULAD_ATTR = ["gender", "region", "age_band"]
OULAD_ATTR_TITLES = {"gender": "Gender", "region": "Region", "age_band": "Age Band"}

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






# selection_rate vs positive_rate dumbbell — DP amplification (RQ1)
# gray = true prevalence, red = model selection rate; gap = amplification

"""
figure 2 : Bias Amplification (selection rate vs. true prevalence)
gray = positive_rate (true prevalence), red = selection_rate
gap = how far the model pushes a group's flag rate above its real base rate

Metric carried: Demographic Parity. Within one attribute block, the spread of
red dots across groups = the DP gap. The gray->red distance = amplification:
the model does not just reflect prevalence, it inflates it.

Finding:

1. LR and XGB amplify almost every group (red right of gray), and the amount
   is uneven -> this is what manufactures the DP gap.
   -* LR income: Above threshold is lifted by only 12 points (0.10 -> 0.22),
     Low income by 60 points (0.21 -> 0.81). The poorer the group, the larger
     the absolute push, ~5x more for Low income. A true-prevalence gap of under
     0.11 is stretched into a selection-rate gap of 0.59 (0.81 - 0.22). 
     The DP gap is created, not inherited. 
     the model learns "low income -> positive" and over-applies it.
   - XGB shows the same direction, smaller magnitude: High +9 points,
     Low +45 points.
   - LR/XGB race: the model lifts every group EXCEPT Asian. Under LR the push
    is largest for Other Race (+37, n=67) and Hispanic (+35), smaller for
    White (+20); XGB shows the same ranking at lower magnitude (+27 to +15).
    Asian is the exception and is covered in point 3.

2. RF barely amplifies anyone (red sits on top of gray, amp ~ 0.8-0.9x).
   This looks fair (small red-dot spread = small DP gap) but it is false
   fairness: RF reaches low disparity by under-flagging EVERYONE, not by
   being even-handed. Same root cause as the RF low-recall result in fig 1.
   -> low DP gap does not always mean fair

3. Suppression, not amplification, for the smallest group.
   Asian (n=54): LR/XGB push it BELOW its true prevalence (0.11 -> 0.06,
   -0.05), and RF flags zero Asians (sel = 0.00). The direction flips for the
   group with the least data. Small n -> unstable, easily suppressed.
   This is a counter-example to "models amplify": amplification is
   group-dependent, not a global threshold shift.

4. The reference dot matters. Reading red dots alone would call Low income
   "high risk by the model"; adding the gray prevalence dot shows the model
   over-states it ~4x. DP measured against selection rate alone hides this;
   the prevalence anchor is what turns a DP gap into an amplification claim.

Why this figure and not a DP bar chart?
A DP difference number says "the gap exists". The dumbbell says "the gap is
manufactured" - the model takes a modest prevalence difference and stretches
it. That is the stronger, harder-to-rebut RQ1 claim, because it cannot be
explained away by "the groups just have different base rates".
"""
C_BASE = "#888888"   # true positive rate (prevalence)
C_SEL = "#d62728"    # model selection rate
ATTR_TITLES_SHORT = {"gender": "Gender", "income": "Income", "race": "Race"}
def f1_baseline_details_dumbbell():
    """
    Within one attribute block, the spread of red dots = DP gap; the differential
    in gray-to-red distance across groups = how that DP gap is amplified.
    """
    ieee_style()
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 4.6), sharex=True)

    # stack all attribute groups vertically, blank gap between attributes
    rowlabels, ypos, sep, layout = [], [], [], {}
    y = 0
    for ai, attr in enumerate(ATTR):
        for g in GROUP_ORDER[attr]:
            layout[(attr, g)] = y
            rowlabels.append(g)
            ypos.append(y)
            y += 1
        if ai < len(ATTR) - 1:
            sep.append(y - 0.5)
            y += 0.6

    for ci, model in enumerate(MODEL_ORDER):
        ax = axes[ci]
        for attr in ATTR:
            sub = detail_df[
                (detail_df["sensitive_attr"] == attr) & (detail_df["model"] == model)
            ].set_index("group")
            for g in GROUP_ORDER[attr]:
                yy = layout[(attr, g)]
                base = sub.loc[g, "positive_rate"]
                sel = sub.loc[g, "selection_rate"]
                ax.plot([base, sel], [yy, yy], color="#cccccc", lw=1.3, zorder=1)
                ax.scatter(base, yy, s=34, color=C_BASE, zorder=3,
                           edgecolor="white", linewidth=0.5)
                ax.scatter(sel, yy, s=34, color=C_SEL, zorder=3,
                           edgecolor="white", linewidth=0.5)
        for s in sep:
            ax.axhline(s, color="#e5e5e5", lw=0.8, zorder=0)
        ax.set_title(MODEL_SHORT[model])
        ax.set_xlim(-0.02, 1.0)
        ax.set_xlabel("Rate")
        ax.invert_yaxis()
        ax.set_yticks(ypos)
        ax.set_yticklabels(rowlabels if ci == 0 else [])

    # attribute names on the far-left outer margin
    for attr in ATTR:
        ys = [layout[(attr, g)] for g in GROUP_ORDER[attr]]
        axes[0].text(-0.46, np.mean(ys), ATTR_TITLES_SHORT[attr],
                     rotation=90, va="center", ha="center",
                     fontsize=9, fontweight="bold")
    
    legend = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_BASE,
               markersize=7, label="True prevalence (positive rate)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_SEL,
               markersize=7, label="Model selection rate"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=2,
               frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Model selection rate vs. true prevalence  "
                 "(gap = bias amplification)", y=1.0, fontsize=10)
    fig.tight_layout(rect=[0.04, 0.02, 1, 0.97])
    save(fig, "f1_baseline_details_dumbbell")





# TPR AND FPR per group (with sample size n), see exactly Who is advantaged/disadvantaged
"""
figure 3: Baseline per-group rates (top: TPR / sensitivity, bottom: FPR)
Finding (RQ1 baseline disparity):

Gender:
1. Female TPR > Male TPR across all three models (LR 0.71 vs 0.55, XGB 0.66 vs 0.53).
   The model catches depression in women more often than in men at the same threshold.
2. Female FPR also higher (LR 0.37 vs 0.26).

Income (the strongest disparity):
1. *TPR rises monotonically as income falls: Above threshold 0.48 < Near poverty 0.77
   < Low income 0.95 (LR). Same ordering in XGB (0.46 / 0.73 / 0.88).
2. *But FPR rises same time: Above threshold 0.19 -> Low income 0.77 (LR).
   The low-income group is not better served, the model just predicts positive far
   more for them. High TPR here is bought with a high false-alarm rate.

Race:
1. Asian is fully degenerate under RF: TPR=0.00, FPR=0.00 — predicts negative for the entire
   group, no positive predictions at all. Small group + class imbalance = total miss.
2. Hispanic and Other Race carry the highest LR TPR (0.81, 0.80), White and Black
   lower (0.63, 0.65). Up to 0.18 spread within one model = clear EOpp gap.
3. FPR ordering tracks TPR again (Hispanic 0.46, White 0.27 under LR), confirming
   the disparity is allocation of positive predictions, not differential skill.
3. FPR moves in the SAME direction as TPR across race groups (Hispanic 0.81/0.46,
    White 0.63/0.27 under LR). A group that is simply predicted MORE accurately
    would show high TPR with LOW FPR; here both rise together, so the higher TPR
    is not better skill on that group.

"""
def r2_baseline_detais_tpr_fpr():
    ieee_style()
    fig, axes = plt.subplots(2, 3, figsize=(8.2, 5.2))
    for r, metric in enumerate(['true_positive_rate', 'false_positive_rate']):
        for c, attr in enumerate(ATTR):
            ax = axes[r, c]
            sub = detail_df[detail_df['sensitive_attr']==attr]
            piv = sub.pivot(index='group', columns='model', values=metric).reindex(GROUP_ORDER[attr])
            piv = piv[MODEL_ORDER]; piv.columns = [MODEL_SHORT[m] for m in MODEL_ORDER]
            ylabels = [f"{g}\n(n={group_n(attr,g)})" for g in piv.index]
            sns.heatmap(piv, annot=True, fmt='.2f', cmap= HEATMAP_CMAP, ax=ax, vmin=0, vmax=1,
                        linewidths=0.5, linecolor='white', cbar=(c==2),
                        cbar_kws={'shrink':0.7} if c==2 else {}, yticklabels=ylabels)
            ax.set_title(ATTR_TITLES[attr] if r==0 else '')
            ax.set_ylabel(('TPR' if r==0 else 'FPR')+'  by group' if c==0 else '')
            ax.set_xlabel(''); ax.tick_params(axis='y', rotation=0)
    fig.suptitle('Baseline per-group rates  (top: TPR / sensitivity,  bottom: FPR)', y=1.01)
    fig.tight_layout()
    save(fig, "r2_baseline_detais_tpr_fpr")



# Full 3x3 fairness grid (all 3 metrics x 3 attrs)
"""
figure 4:  Fairness Metric Grid — All 45 Experiments (3 models × 3 attributes × 5 conditions)
Layout: 3×3 grid, rows = sensitive attribute (Gender / Income / Race), columns = fairness metric (DP / EOpp / EOdds)

Key findings by attribute:

Gender (top row):
1. Baseline unfairness is moderate: LR has highest gap (DP=0.14, EOpp/EOdds≈0.16), XGB moderate (0.07/0.13), RF lowest DP (0.05) but EOpp still 0.17
2. ALL four methods reduce LR gender bias well — every bar below 0.10 threshold
3. XGB gender: Suppression achieves near-zero DP (0.009), ThresholdOptimizer near-zero DP (0.01) but EOpp increases to 0.12
4. RF gender: ThresholdOptimizer is the only method that substantially reduces EOpp (0.17→0.05), but at cost of accuracy drop (0.83→0.65)

Income (middle row):
1. Baseline disparity is SEVERE across all metrics and models: DP 0.32–0.59, EOpp 0.28–0.47, EOdds 0.30–0.58
2. *LR+income+EG is degenerate (recall=0.0, all-negative predictions) → marked with red ×. The near-zero fairness gap is "false fairness"
3. LR+income+TO: flips to aggressive positive prediction (recall=0.81, accuracy drops to 0.46), still achieves moderate fairness improvement
4. *Reweighing consistently reduces income gap by ~60-70% across all models (most stable method for income)
5. *Suppression barely helps income — gap remains large (e.g. LR DP: 0.59→0.45, XGB DP: 0.48→0.38). Income proxies persist in other features

Race (bottom row):
1. Baseline EOpp/EOdds extremely high (0.45–0.68) across all models — race is the hardest attribute to mitigate
2. EG achieves best DP reduction for race (XGB: 0.38→0.09, LR: 0.47→0.06) but EOpp remains >0.16 
3. ThresholdOptimizer: best DP for XGB+race (0.38→0.02) but EOpp stays at 0.63 — same pattern as EG
4. NO method reduces race EOpp/EOdds below 0.10 — structural difficulty, likely due to small subgroup sizes (Asian, Other)
5. *Suppression sometimes increases race disparity (RF: 0.24→0.23 DP flat, EOpp 0.45→0.50 worse)

summary:
- Gender is the easiest attribute to mitigate (most methods push below 0.10)
- Income is the most responsive to Reweighing but the hardest to fix via Suppression (proxy features)
"""
def r3_fairness_mit_grid_full():
    ieee_style()
    fig, axes = plt.subplots(3, 3, figsize=(7.16, 7.5))
    for ri, attr in enumerate(ATTR):
        for ci, fm in enumerate(FAIR_METRICS):
            ax = axes[ri, ci]; sub = mit_df[mit_df['sensitive_attr']==attr]
            x = np.arange(len(MODEL_ORDER)); w = 0.15; n = len(METHOD_ORDER)
            for mi, method in enumerate(METHOD_ORDER):
                vals = [sub[(sub['model']==md)&(sub['miti_method']==method)][fm].values[0]
                        for md in MODEL_ORDER]
                off = (mi - n/2 + 0.5)*w
                ax.bar(x+off, vals, w, color=METHOD_COLORS[method], edgecolor='white', linewidth=0.3)
                # flag degenerate (recall<0.01) cells with a small x
                for j, md in enumerate(MODEL_ORDER):
                    rr = sub[(sub['model']==md)&(sub['miti_method']==method)]
                    if len(rr) and rr['recall'].values[0] < 0.01:
                        ax.text(x[j]+off, 0.005, '×', ha='center', va='bottom',
                                fontsize=7, color='red', fontweight='bold')
            ax.set_xticks(x); ax.set_xticklabels([MODEL_SHORT[m] for m in MODEL_ORDER])
            ax.set_ylim(0, max(sub[fm].max()*1.15, 0.12))
            ax.axhline(0.1, color='red', ls='--', lw=0.7, alpha=0.7)
            if ci==0: ax.set_ylabel(ATTR_TITLES[attr], fontweight='bold')
            if ri==0: ax.set_title(fm, fontsize=8.5)
            ax.spines[['top','right']].set_visible(False)
    h = [Patch(facecolor=METHOD_COLORS[m], edgecolor='white') for m in METHOD_ORDER]
    h += [Line2D([0],[0], color='red', ls='--', lw=0.7)]
    lab = [METHOD_SHORT[m] for m in METHOD_ORDER] + ['threshold 0.1 (heuristic)']
    fig.legend(h, lab, loc='lower center', ncol=6, bbox_to_anchor=(0.5,-0.03), fontsize=7.5)
    fig.text(0.5, -0.055, '×  = degenerate model (recall ≈ 0, e.g. LR+income+EG collapse)',
             ha='center', fontsize=7, color='red')
    fig.tight_layout(rect=[0,0.04,1,1])
    save(fig, "r3_fairness_mit_grid_full")




"""
figure 5: Performance under Mitigation (Accuracy / Recall / F1, 3×3 grid)


Row 1 (Accuracy):
- Gender column: all methods land within ~2% of baseline minimal cost, except TO under RF
- Income column: LR+EG orange bar hits 0.87 — HIGHEST accuracy, but recall=0 . Pure majority-class prediction, "fake fairness"
- TO (green) drops accuracy the most everywhere (RF+income: 0.83→0.51),
  expected because TO shifts the threshold to catch more positives

Row 2 (Recall):
- RF baseline recall ≈ 0.30 across all attrs, because the threshold under 13% imbalance, (AUC is fine)
- TO is the only method that systematically boosts recall:
  RF+gender 0.30→0.70, RF+income 0.30→0.63, LR+income 0.66→0.81
- LR+income+EG: degenerate
- EG on XGB holds up (recall stays 0.53–0.62) , XGB stronger than LR

Row 3 (F1):
- All F1 scores compressed in 0.25–0.39 range due to class imbalance
- XGB+gender: EG gets the best F1 (0.38) with no recall loss, but DP 0.07→0.02, EOpp 0.13→0.07, cleanest win
- LR+income+EG: F1=0, confirms degenerate

 Key findings:

1. Accuracy is misleading under imbalance
2. TO = recall booster, accuracy costs predictable
   Medical context: missing depression cases > false alarms, so the tradeoff
   may be justified.
3. EG is base-model-capacity-sensitive
   Works on XGB (enough flexibility to satisfy constraints + classify).
   Collapses on LR when baseline bias is large
   linear boundary cannot satisfy tight fairness constraints.
4. Gender is easy, income/race are hard
"""
def r4_performance_mit_grid_full():
    ieee_style()
    metrics = [('accuracy', 'Accuracy'), ('recall', 'Recall'), ('f1', 'F1-Score')]
    fig, axes = plt.subplots(3, 3, figsize=(7.16, 6.5), sharey='row')

    for ri, (pm, pl) in enumerate(metrics):
        for ci, attr in enumerate(ATTR):
            ax = axes[ri, ci]
            sub = mit_df[mit_df['sensitive_attr'] == attr]
            x = np.arange(len(MODEL_ORDER))
            w = 0.15
            n = len(METHOD_ORDER)

            for mi, method in enumerate(METHOD_ORDER):
                vals = [sub[(sub['model'] == md) & (sub['miti_method'] == method)][pm].values[0]
                        for md in MODEL_ORDER]
                off = (mi - n / 2 + 0.5) * w
                bars = ax.bar(x + off, vals, w,
                              color=METHOD_COLORS[method],
                              edgecolor='white', linewidth=0.3)

                # --- degenerate annotation: recall < 0.01 ---
                for bi, (bar, md) in enumerate(zip(bars, MODEL_ORDER)):
                    row = sub[(sub['model'] == md) & (sub['miti_method'] == method)]
                    if row['recall'].values[0] < 0.01:
                        cx = bar.get_x() + bar.get_width() / 2
                        cy = bar.get_height()
                        ax.plot(cx, cy + 0.02, 'x', color='red',
                                markersize=5, markeredgewidth=1.5, zorder=5)
                        if ri == 0 and pm == 'accuracy':
                            ax.text(cx, cy + 0.05, 'recall\n≈ 0',
                                    ha='center', va='bottom', fontsize=5,
                                    color='red', fontweight='bold')

            ax.set_xticks(x)
            ax.set_xticklabels([MODEL_SHORT[m] for m in MODEL_ORDER])
            if ci == 0:
                ax.set_ylabel(pl)
            if ri == 0:
                ax.set_title(ATTR_TITLES[attr])
            ax.spines[['top', 'right']].set_visible(False)

    axes[0, 0].set_ylim(0, 1.0)
    axes[1, 0].set_ylim(0, 1.0)
    axes[2, 0].set_ylim(0, 0.5)

    h = [Patch(facecolor=METHOD_COLORS[m], edgecolor='white') for m in METHOD_ORDER]
    fig.legend(h, [METHOD_SHORT[m] for m in METHOD_ORDER],
               loc='lower center', ncol=5,
               bbox_to_anchor=(0.5, -0.02), fontsize=8)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    save(fig, "r4_performance_mit_grid_full")

"""
Figure 6&7: Equal Opportunity — Baseline vs Best Mitigation 

Best mitigation selected per model × attribute (recall > 0.05 filter applied):

  Gender:
    LR  → Suppression        (EOpp 0.0158)
    RF  → ThresholdOptimizer  (EOpp 0.0547)
    XGB → Reweighing          (EOpp 0.0156)

  Income:
    LR  → ThresholdOptimizer  (EOpp 0.0735)  [EG excluded: recall=0, degenerate]
    RF  → ExponentiatedGrad.  (EOpp 0.0823)
    XGB → Reweighing          (EOpp 0.0808)

  Race:
    LR  → ExponentiatedGrad.  (EOpp 0.1613)
    RF  → ExponentiatedGrad.  (EOpp 0.3077)
    XGB → ExponentiatedGrad.  (EOpp 0.6282)

Findings:

1. Gender is the easiest to fix. All three baselines already small
   (0.13-0.17), all drop below the 0.1 heuristic after mitigation.
   No single method dominates — Suppression, TO, Reweighing each
   win once, suggesting gender bias is shallow and method-agnostic.

2. Income shows the largest absolute reduction. LR baseline 0.47 drops
   to 0.07 (TO), XGB 0.42 drops to 0.08 (Reweighing). 

3. Race is the hardest to mitigate. EG wins all three models, but the
   outcome varies wildly: LR 0.64→0.16 (good), RF 0.45→0.31 (partial),
   XGB 0.68→0.63 (nearly unchanged). 


4. *Mitigation difficulty depends on baseline disparity magnitude, subgroup
   sample sizes, and model complexity 
   - Gender: small baseline gaps (0.13-0.17) + balanced group sizes
     → easy to fix regardless of method.
   - Income: large baseline gaps (0.42-0.47) but sufficient per-group
     samples → large absolute reduction achievable.
   - Race: similar prevalence across groups, yet models manufacture
     large TPR gaps (amplification artifact). 5 groups with two very
     small subsamples (Asian n=54, Other Race n=67) make TPR alignment
     unstable. 
5. No method is universally best. Reweighing wins 2 cells, EG wins 4,
   TO wins 2, Suppression wins 1. The optimal choice depends on the
   model-attribute combination, not the method alone.

"""
def r5_disparity_reduction():
    ieee_style()
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.7), sharey=True)
    for idx, attr in enumerate(ATTR):
        ax = axes[idx]; sub = mit_df[mit_df['sensitive_attr']==attr]
        x = np.arange(len(MODEL_ORDER)); w = 0.7
        base_vals, best_vals = [], []
        for md in MODEL_ORDER:
            s = sub[sub['model']==md]
            base = s[s['miti_method']=='Baseline']['Equal Opportunity'].values[0]
            cand = s[(s['recall']>0.05) & (s['miti_method']!='Baseline')]
            best = cand['Equal Opportunity'].min() if len(cand) else base
            base_vals.append(base); best_vals.append(best)
        ax.bar(x, base_vals, w, color='#bdc3c7', label='Baseline', edgecolor='white')
        ax.bar(x, best_vals, w*0.55, color='#27ae60', label='Best mitigation', edgecolor='white')
        ax.set_xticks(x); ax.set_xticklabels([MODEL_SHORT[m] for m in MODEL_ORDER])
        ax.axhline(0.1, color='red', ls='--', lw=0.7, alpha=0.6)
        ax.set_title(ATTR_TITLES[attr]); ax.spines[['top','right']].set_visible(False)
        if idx==0: ax.set_ylabel('Equal Opportunity Diff.')
    axes[0].legend(loc='upper right', fontsize=7, framealpha=0.9)
    fig.suptitle('Equal Opportunity: Baseline vs Best Mitigation', y=1.03)
    fig.tight_layout()
    save(fig,  "r5_disparity_reduction")

# fig7
def r6_disparity_reduction():
    ppt_style()
    fig, ax = plt.subplots(figsize=(11, 6))
    rows = []
    for attr in ATTR:
        for md in MODEL_ORDER:
            s = mit_df[(mit_df['sensitive_attr']==attr)&(mit_df['model']==md)]
            base = s[s['miti_method']=='Baseline']['Equal Opportunity'].values[0]
            cand = s[(s['recall']>0.05)&(s['miti_method']!='Baseline')]
            if len(cand):
                bi = cand['Equal Opportunity'].idxmin()
                best = cand.loc[bi,'Equal Opportunity']; bm = METHOD_SHORT[cand.loc[bi,'miti_method']]
            else:
                best, bm = base, '—'
            rows.append((f"{ATTR_TITLES[attr]}\n{MODEL_SHORT[md]} ({bm})", base, best))
    rows = rows[::-1]
    y = np.arange(len(rows))
    for i,(lab,b,a) in enumerate(rows):
        ax.plot([a,b],[i,i], color='#bbb', lw=3, zorder=1)
        ax.scatter(b, i, s=200, color='#2c3e50', zorder=2)
        ax.scatter(a, i, s=200, color='#27ae60', zorder=2)
    ax.axvline(0.1, color='red', ls='--', lw=1.5, alpha=0.7)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=11)
    ax.set_xlabel('Equal Opportunity Diff.'); ax.set_xlim(-0.02, 0.72)
    ax.set_title('Baseline → Best Mitigation (per model × attribute)')
    ax.spines[['top','right']].set_visible(False)
    ax.legend(handles=[Line2D([0],[0],marker='o',color='w',markerfacecolor='#2c3e50',markersize=13,label='Baseline'),
                       Line2D([0],[0],marker='o',color='w',markerfacecolor='#27ae60',markersize=13,label='Best mitigation'),
                       Line2D([0],[0],color='red',ls='--',lw=1.5,label='0.1 heuristic')],
              loc='upper right', fontsize=12)
    fig.tight_layout()
    save(fig, "r6_disparity_reduction")


# fig8
"""
figure 8: F1 vs Equal Opportunity Tradeoff Scatter ( RQ3)

x = F1-Score, y = EOpp Difference
3 panels: Gender / Income / Race
red × = degenerate model (recall < 0.05, all-negative predictions)
red dashed line = 0.1 fairness threshold

Why F1 not Accuracy on x-axis?
Class imbalance (~13% positive). Accuracy is dominated by majority class,
so a recall-collapsed model (predict all negative) gets accuracy=0.87
and lands in the "ideal" bottom-right corner. F1 correctly puts it at
the left (F1=0) because F1 penalizes zero recall.

Ideal region: bottom-right (high F1 + low disparity)
Below baseline = more fair. Left of baseline = performance cost.

Gender panel:
1. all 15 points squeezed in F1≈0.31-0.38, EOpp≈0.01-0.17
   → tight cluster, small range both axes
2. most mitigation points sit BELOW baseline AND keep similar F1
   → essentially free lunch, no real tradeoff
3. no single method dominates — RW/EG/TO/Supp all have winners
   → gender bias is shallow, method-agnostic to fix

Income panel:
1. ★ DEGENERATE POINT bottom-left: LR+income+EG, F1=0, EOpp=0
   → false fairness. model predicts all negative, recall=0.
   trivially zero disparity because no one gets predicted positive.
   THIS is why we use F1 not Accuracy — Accuracy would put this at 0.87.
2. baselines cluster at top (EOpp 0.28-0.47), mitigation pulls down
   but costs F1: e.g. LR+TO drops EOpp 0.47→0.07 but F1 0.35→0.29
   XGB+RW drops EOpp 0.42→0.08, F1 only 0.37→0.35 (cheaper)
   → tradeoff exists, but cost varies by model×method
3. Suppression (purple): all 3 models move down (EOpp drops 0.12-0.16)
   with no F1 cost — partial improvement, but still worse than
   RW/EG/TO which achieve EOpp < 0.10. Suppression reduces income
   bias but does not solve it.
Race panel:
1. ★ ALL points above 0.1 threshold line. most above 0.3, many near 0.7
   → race EOpp is essentially unmitigable with current methods
2. XGB+race: all 5 methods stuck at EOpp 0.63-0.68, zero movement
   → hardest cell in entire 3×3 grid, nothing works
3. only LR+EG gets close to acceptable (EOpp 0.16) but F1 crashes
   0.35→0.15 — you pay massive performance for marginal fairness
4. Suppression backfires on race: RF+Supp EOpp=0.50 > baseline 0.45
    → removing race feature makes it WORSE (proxy encoding)

Cross-panel patterns:
1. tradeoff severity: gender ≈ free lunch, income = moderate,
    race = money can not buy fairness
2. no universal best method. RW good on gender/income,
3. Suppression is the worst method overall — never best on income
    or race, sometimes backfires. confirms naive suppression is not
    a real mitigation strategy
4. degenerate model root cause: LR linear capacity too weak to
    satisfy tight EG fairness constraint when baseline bias is huge
    (income EOpp≈0.47). not a group-count issue — it is model
    capacity × constraint tightness × baseline disparity magnitude
"""
def f2_tradeoff_f1_EOpp():
    ieee_style()
    markers = {'LogisticRegression':'o','RandomForest':'s','XGB':'^'}
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.9), sharey=True)
    for idx, attr in enumerate(ATTR):
        ax = axes[idx]; sub = mit_df[mit_df['sensitive_attr']==attr]
        for _, r in sub.iterrows():
            degen = r['recall'] < 0.01            # collapsed / degenerate model
            ax.scatter(r['f1'], r['Equal Opportunity'],
                       c='none' if degen else METHOD_COLORS[r['miti_method']],
                       edgecolors=METHOD_COLORS[r['miti_method']] if degen else 'white',
                       marker=markers[r['model']], s=48,
                       linewidth=1.1 if degen else 0.4, alpha=0.9, zorder=3)
            if degen:
                ax.scatter(r['f1'], r['Equal Opportunity'], marker='x', c='red',
                           s=34, linewidth=1.1, zorder=4)
        ax.axhline(0.1, color='red', ls='--', lw=0.7, alpha=0.6)
        ax.set_xlabel('F1-Score'); ax.set_title(ATTR_TITLES[attr]); ax.set_xlim(-0.02, 0.45)
        if idx==0: ax.set_ylabel('Equal Opportunity Diff.')
        ax.spines[['top','right']].set_visible(False)
    # TWO-ROW legend: row1 methods, row2 models (+ degenerate marker)
    mh = [Line2D([0],[0],marker='o',color='w',markerfacecolor=METHOD_COLORS[m],markersize=7,
                 label=METHOD_SHORT[m]) for m in METHOD_ORDER]
    kh = [Line2D([0],[0],marker=markers[m],color='w',markerfacecolor='gray',markersize=7,
                 label=MODEL_SHORT[m]) for m in MODEL_ORDER]
    kh += [Line2D([0],[0],marker='x',color='red',lw=0,markersize=7,
                  label='degenerate model')]
    leg1 = fig.legend(handles=mh, loc='lower center', bbox_to_anchor=(0.5,-0.07), ncol=5, fontsize=7.5)
    fig.add_artist(leg1)
    fig.legend(handles=kh, loc='lower center', bbox_to_anchor=(0.5,-0.16), ncol=4, fontsize=7.5)
    fig.tight_layout(rect=[0,0.08,1,1])
    save(fig, "f2_tradeoff_f1_EOpp")

# TODO:  进行cross domain comparison的作图
METHOD_MARKERS = {
    "Reweighing": "s",
    "ExponentiatedGradient": "D",
    "ThresholdOptimizer": "^",
    "Suppression": "v",
}

# fig 9
"""
figure 9: Method-wise EOpp Gap Comparison (RQ3)

y = Equal Opportunity Difference (TPR gap between worst and best group)
3 panels: Gender / Income Group / Race Group
grey bar = baseline, colored markers = 4 mitigation methods
red × = degenerate (recall < 0.05), red dashed = 0.1 fairness threshold

What this figure answers:
Which mitigation method reduces EOpp most, per model × attribute?
Unlike F2 (best-only), this shows ALL 4 methods side by side so you
can compare method effectiveness directly.

Gender panel:
1. baseline EOpp is low to begin with (0.13–0.17), all three models
2. every method pulls below baseline for every model
   → gender bias is easy to mitigate, all methods work
3. most points land below 0.1 threshold
   → gender is essentially "solved" by any method
4. no clear winner — RW, EG, TO, Supp all achieve EOpp < 0.05
   in at least one model. method choice barely matters here

Income panel:
1. ★ LR+EG = red ×, EOpp=0.00 — DEGENERATE. recall=0, F1=0.
   model predicts all negative. trivially zero disparity.
   false fairness, not real mitigation.
2. baselines are high: LR=0.47, RF=0.28, XGB=0.42
   → income bias is structurally embedded
3. RW and EG (non-degenerate cases) pull EOpp below 0.10 for
   RF and XGB — genuine improvement
4. TO is inconsistent: good on LR (0.47→0.07) but bad on
   RF (0.28→0.25) and XGB (0.42→0.23) — post-processing
   struggles when base model already has low recall (RF)
5. Suppression barely moves the needle: LR 0.47→0.35,
   XGB 0.42→0.28, RF improves slightly 0.28→0.12
   → income-correlated proxy features make suppression
   ineffective. removing the income column does not remove
   income signal from the data

Race panel:
1. ★ ALL points above 0.1 except LR+EG (0.16) — race is the
   hardest attribute to mitigate by a wide margin
2. XGB+race: all 4 methods stuck at 0.63–0.68, basically
   equal to baseline 0.68 → nothing works, zero movement
3. RF+race: RW does nothing (0.45→0.45), EG partial (0.31),
   TO partial (0.33), Supp makes it WORSE (0.45→0.50)
4. LR+race: EG is the only method that gets close (0.64→0.16)
   but at massive F1 cost (0.35→0.15) — check F2 for the
   performance tradeoff
5. Suppression backfires: RF 0.45→0.50, XGB 0.68→0.68
   → race feature removal activates proxy encoding,
   model reconstructs race signal from correlated features

Cross-panel takeaways:
1. difficulty gradient: gender (trivial) → income (moderate)
   → race (near-impossible with current methods)
2. no universal best method — effectiveness is
   attribute × model specific
3. Suppression is consistently the weakest or worst method,
   sometimes increasing disparity — naive feature removal
   is not a valid mitigation strategy
4. degenerate models are a real risk: EG under tight constraints
   + weak model capacity (LR) + high baseline bias (income 0.47)
   = model collapse to all-negative predictions
5. race disparity is robust to all four mitigation approaches,
   especially for complex models (XGB) — suggests the
   disparity source is not simple feature-level bias but
   deeper distributional differences across racial groups
"""
def f3_method_gap_comparison():
    """All-method EOpp comparison — reads directly from mit_df."""
    ieee_style()

    gap_rows = []
    for _, row in mit_df.iterrows():
        gap_rows.append({
            "model": row["model"],
            "attr": row["sensitive_attr"],
            "method": row["miti_method"],
            "abs_gap": row["Equal Opportunity"],
            "degenerate": row["recall"] < 0.05,
        })
    gaps = pd.DataFrame(gap_rows)

    fig, axes = plt.subplots(1, 3, figsize=(7.16, 3.2), sharey=True)

    for idx, attr in enumerate(ATTR):
        ax = axes[idx]

        for mi, model in enumerate(MODEL_ORDER):
            x_center = mi

            base_gap = gaps[
                (gaps["attr"] == attr)
                & (gaps["model"] == model)
                & (gaps["method"] == "Baseline")
            ]["abs_gap"].values[0]

            ax.plot(
                [x_center - 0.35, x_center + 0.35],
                [base_gap, base_gap],
                color="#bdc3c7", lw=2.5, zorder=1,
            )

            offsets = np.linspace(-0.2, 0.2, len(METHOD_MARKERS))
            for ji, method in enumerate(METHOD_MARKERS.keys()):
                row = gaps[
                    (gaps["attr"] == attr)
                    & (gaps["model"] == model)
                    & (gaps["method"] == method)
                ]
                if len(row) == 0:
                    continue
                r = row.iloc[0]
                xp = x_center + offsets[ji]

                ax.plot(
                    [xp, xp], [base_gap, r["abs_gap"]],
                    color=METHOD_COLORS[method], lw=1.2, alpha=0.6, zorder=2,
                )

                if r["degenerate"]:
                    ax.scatter(
                        xp, r["abs_gap"], s=40, facecolors="white",
                        edgecolors=METHOD_COLORS[method], linewidth=1,
                        marker=METHOD_MARKERS[method], zorder=4,
                    )
                    ax.plot(
                        xp, r["abs_gap"], "x", color="red",
                        markersize=5, markeredgewidth=1.2, zorder=5,
                    )
                else:
                    ax.scatter(
                        xp, r["abs_gap"], s=40, color=METHOD_COLORS[method],
                        edgecolors="white", linewidth=0.4,
                        marker=METHOD_MARKERS[method], zorder=4,
                    )

        ax.axhline(0.1, color="red", ls="--", lw=0.7, alpha=0.5)
        ax.set_xticks(range(len(MODEL_ORDER)))
        ax.set_xticklabels([MODEL_SHORT[m] for m in MODEL_ORDER])
        ax.set_title(ATTR_TITLES[attr])
        ax.spines[["top", "right"]].set_visible(False)
        if idx == 0:
            ax.set_ylabel("TPR Gap (Equal Opportunity Diff.)")

    mh = [Line2D([0], [0], color="#bdc3c7", lw=2.5, label="Baseline")]
    mh += [
        Line2D(
            [0], [0], marker=METHOD_MARKERS[m], color="w",
            markerfacecolor=METHOD_COLORS[m], markersize=7,
            label=METHOD_SHORT[m],
        )
        for m in METHOD_MARKERS
    ]
    mh += [
        Line2D([0], [0], marker="x", color="red", lw=0,
               markersize=6, label="degenerate"),
        Line2D([0], [0], color="red", ls="--", lw=0.7,
               label="0.1 threshold"),
    ]
    fig.legend(
        handles=mh, loc="lower center", ncol=len(mh),
        bbox_to_anchor=(0.5, -0.08), fontsize=6.5,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.98])
    save(fig, "f3_method_gap_comparison")
 


from matplotlib.transforms import blended_transform_factory

"""
    figure 10: TPR-span dumbbell, before vs after best mitigation.
    18 segments = 9 (model × attr) × 2 (baseline + best mitigation).

    Findings:
    1. Gender: baseline gaps small (0.13–0.17), all close to < 0.06 after
    2. Income: largest baseline gaps (0.28–0.47), TO/RW reduce well
    3. Race: most resistant — XGB+EG barely shrinks (0.68→0.63)
    4. Method diversity: gender/income pick varied methods;
       race uniformly selects EG yet EG struggles with 5-group disparity
"""
def f4_mit_detail_gap_dumbbell():
   
    ieee_style()

    # ── best method per model × attr (lowest EOpp, recall ≥ 0.05) ──
    recs = []
    for attr in ATTR:
        for model in MODEL_ORDER:
            cands = mit_df.query(
                "model == @model and sensitive_attr == @attr "
                "and miti_method != 'Baseline' and recall >= 0.05"
            )
            if cands.empty:
                continue
            method = cands.loc[cands["Equal Opportunity"].idxmin(), "miti_method"]

            bg = mit_detail_df.query(
                "model == @model and sensitive_attr == @attr "
                "and miti_method == 'Baseline'"
            )
            ag = mit_detail_df.query(
                "model == @model and sensitive_attr == @attr "
                "and miti_method == @method"
            )

            recs.append(dict(
                model=model, attr=attr, method=method,
                b_hi=bg["true_positive_rate"].max(),
                b_lo=bg["true_positive_rate"].min(),
                a_hi=ag["true_positive_rate"].max(),
                a_lo=ag["true_positive_rate"].min(),
            ))

    # ── x positions: 3 attr groups × 3 models, gaps between groups ──
    GAP = 0.7
    OFF = 0.14
    xs, xi = [], 0
    for ai in range(3):
        for _ in range(3):
            xs.append(xi)
            xi += 1
        if ai < 2:
            xi += GAP

    # ── figure ──
    fig, ax = plt.subplots(figsize=(7.16, 2.6))
    #   BC = "#b0b0b0"      # baseline gray
    #   AC = "#27ae60"       # after — uniform green
    BC = "#c0392b"       # baseline red (problem state)
    AC = "#2980b9"       # after blue (improved)
    MS = 3.5
    LW = 1.8

    for i, r in enumerate(recs):
        x = xs[i]

        # baseline (left)
        ax.plot([x - OFF] * 2, [r["b_lo"], r["b_hi"]],
                color=BC, lw=LW, solid_capstyle="round", zorder=2)
        ax.plot(x - OFF, r["b_hi"], "o", color=BC, ms=MS, zorder=3)
        ax.plot(x - OFF, r["b_lo"], "o", color=BC, ms=MS, zorder=3,
                 mew=1.0)
        # mfc="white",
        # after (right)
        ax.plot([x + OFF] * 2, [r["a_lo"], r["a_hi"]],
                color=AC, lw=LW, solid_capstyle="round", zorder=2)
        ax.plot(x + OFF, r["a_hi"], "o", color=AC, ms=MS, zorder=3)
        ax.plot(x + OFF, r["a_lo"], "o", color=AC, ms=MS, zorder=3,
                 mew=1.0)
        # mfc="white",
    # ── x ticks ──
    ax.set_xticks(xs)
    ax.set_xticklabels([
        f"{MODEL_SHORT[r['model']]}\n({METHOD_SHORT[r['method']]})"
        for r in recs
    ])

    # ── attribute group labels above plot ──
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    for ai, attr in enumerate(ATTR):
        gx = xs[ai * 3: ai * 3 + 3]
        ax.text(np.mean(gx), 1.06, ATTR_TITLES[attr],
                transform=trans, ha="center", va="bottom",
                fontsize=9, fontweight="bold")

    # ── dashed separators ──
    for k in range(2):
        sx = (xs[k * 3 + 2] + xs[k * 3 + 3]) / 2
        ax.axvline(sx, color="#dee2e6", ls=":", lw=0.6, zorder=0)

    # ── axes ──
    ax.set_xlim(xs[0] - 0.5, xs[-1] + 0.5)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("True positive rate")
    ax.spines[["top", "right"]].set_visible(False)

    # ── legend (bottom center) ──
    h = [
        Line2D([], [], color=BC, marker="o", ms=MS, lw=LW, label="Baseline"),
        Line2D([], [], color=AC, marker="o", ms=MS, lw=LW, label="After mitigation"),
        # Line2D([], [], ls="none", marker="o", ms=MS,
        #        mfc="#888", mec="#888", label="Best-served"),
        # Line2D([], [], ls="none", marker="o", ms=MS,
        #        mfc="white", mec="#888", mew=1.0, label="Worst-served"),
    ]
    fig.legend(
        handles=h, loc="lower center",
        bbox_to_anchor=(0.5, -0.06),
        ncol=4, frameon=False, fontsize=7,
        handletextpad=0.4, columnspacing=1.2,
    )

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    save(fig, "f4_mit_detail_gap_dumbbell")




# Cross-domain code
# Shared helper: compute % EOpp reduction per (model, method)
# Filters: recall < 0.05 (degenerate), baseline EOpp < 0.05 (unstable %)
def _eopp_reduction_matrix(df, attrs, recall_thr=0.05, base_thr=0.05):
    methods = ["Reweighing", "ExponentiatedGradient", "ThresholdOptimizer", "Suppression"]
    mat = np.full((len(methods), len(MODEL_ORDER)), np.nan)
    mask_degen = np.zeros_like(mat, dtype=bool)
    for ci, model in enumerate(MODEL_ORDER):
        for ri, method in enumerate(methods):
            reds = []
            is_degen = False
            for attr in attrs:
                base_row = df[
                    (df["model"] == model)
                    & (df["sensitive_attr"] == attr)
                    & (df["miti_method"] == "Baseline")
                ]
                meth_row = df[
                    (df["model"] == model)
                    & (df["sensitive_attr"] == attr)
                    & (df["miti_method"] == method)
                ]
                if len(base_row) == 0 or len(meth_row) == 0:
                    continue
                base_eopp = base_row["Equal Opportunity"].values[0]
                meth_eopp = meth_row["Equal Opportunity"].values[0]
                meth_recall = meth_row["recall"].values[0]
                if meth_recall < recall_thr:
                    is_degen = True
                    continue
                if base_eopp < base_thr:
                    continue
                reds.append((base_eopp - meth_eopp) / base_eopp * 100)
            if is_degen and len(reds) == 0:
                mask_degen[ri, ci] = True
            elif reds:
                mat[ri, ci] = np.mean(reds)
    return mat, mask_degen
 
 

"""
figure 11: Cross-Domain Heatmap — % EOpp Reduction per Model × Method

Each cell = mean % EOpp reduction across sensitive attributes for that model+method combo.
Positive = improvement (green/yellow), negative = backfire (dark purple).
Baseline EOpp < 0.05 attributes skipped (nothing to reduce).
Degenerate models (recall < 0.05) excluded and marked ×.

★ NHANES all positive (every cell +20% or above).
  OULAD mixed: only EG consistently positive, other 3 methods near-zero or backfire.

# NHANES(left panel):
2. LR benefits most: EG +76%, TO +74%, RW +53%
   → linear boundary is easiest to constrain/adjust
3. EG has the highest floor (+40%) and ceiling (+76%)
   → strongest single method, but variable across models (range 36pp)
4. Suppression most stable across models (range only 7pp) but lowest ceiling (+39%)
   → helps everywhere, solves nowhere
5. TO most variable: LR +74% vs XGB +20% (range 54pp)
   → highly dependent on base model, not blindly transferable   


OULAD (right panel):
1. RW: all three models negative (-3%, -9%, -4%)
   → reweighing consistently backfires in education domain.
   OULAD baseline bias is already mild, reweighing overshoots
2. EG: the only method with meaningful positive numbers
   (LR +33%, RF +31%), but XGB only +4%
   → in-processing constraint is the only thing that transfers
3. ★ RF+TO = -236%. catastrophic backfire.
   this is the single most extreme cell in the entire figure.
   ThresholdOptimizer on RF amplifies disparity instead of reducing it.
4. Suppression: mixed bag (LR +6%, RF -15%, XGB +14%)
   → no clear direction, confirms suppression is unreliable

Cross-domain patterns:
1. ★ the two domains have fundamentally different fairness landscapes.
   NHANES has deep structural bias (race EOpp up to 0.68, income 0.47)
   that methods can meaningfully reduce.
   OULAD has mild baseline bias — mitigation methods often overshoot
   or have nothing meaningful to fix, leading to near-zero or negative results.
2. EG is the only method that works in both domains.
   RW transfers worst (strong in NHANES, backfires in OULAD).
   TO is domain-dependent and unstable (strong in NHANES, catastrophic on OULAD RF).
3. RF is the problematic model in OULAD: TO -236%, Supp -15%, RW -9%.
   only EG (+31%) saves it. in NHANES, RF is middle-of-the-road, no disasters.
   → same model, opposite behavior across domains
4. practical takeaway: mitigation methods designed for high-bias settings
   (healthcare) do not safely transfer to low-bias settings (education).
   applying them blindly can make fairness worse, not better.
"""
def x1_crossdomain_heatmap():
    ieee_style()
    methods = ["Reweighing", "ExponentiatedGradient", "ThresholdOptimizer", "Suppression"]
    m_short = [METHOD_SHORT[m] for m in methods]
 
    nh_mat, nh_degen = _eopp_reduction_matrix(mit_df, ATTR)
    ou_mat, ou_degen = _eopp_reduction_matrix(oulad_df, OULAD_ATTR)
 
    fig, axes = plt.subplots(
        1, 2, figsize=(7.16, 2.8), sharey=True,
        gridspec_kw={"wspace": 0.08, "right": 0.88},
    )
 
    CLIP = 100  # color scale range; raw value still shown in annotation
 
    for ax, mat, degen, title in [
        (axes[0], nh_mat, nh_degen, "NHANES (Healthcare)"),
        (axes[1], ou_mat, ou_degen, "OULAD (Education)"),
    ]:
        display = np.clip(mat, -CLIP, CLIP)
        masked = np.ma.masked_invalid(display)
        im = ax.imshow(masked, cmap="viridis", vmin=-CLIP, vmax=CLIP, aspect="auto")
 
        for ri in range(len(methods)):
            for ci in range(len(MODEL_ORDER)):
                if degen[ri, ci]:
                    ax.text(
                        ci, ri, "×", ha="center", va="center",
                        fontsize=9, fontweight="bold", color="#888888",
                    )
                elif np.isnan(mat[ri, ci]):
                    ax.text(
                        ci, ri, "—", ha="center", va="center",
                        fontsize=7, color="#bbbbbb",
                    )
                else:
                    v = mat[ri, ci]
                    abs_v = min(abs(v), CLIP)
                    color = "white" if abs_v > CLIP * 0.55 else "black"
                    sign = "+" if v > 0 else ""
                    ax.text(
                        ci, ri, f"{sign}{v:.0f}%", ha="center", va="center",
                        fontsize=7.5,
                        fontweight="bold" if abs(v) > 100 else "normal",
                        color=color,
                    )
 
        ax.set_xticks(range(len(MODEL_ORDER)))
        ax.set_xticklabels([MODEL_SHORT[m] for m in MODEL_ORDER])
        ax.set_title(title, fontsize=9, pad=6)
        ax.tick_params(length=0)
 
    axes[0].set_yticks(range(len(methods)))
    axes[0].set_yticklabels(m_short)
 
    cax = fig.add_axes([0.90, 0.18, 0.015, 0.65])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("EOpp reduction (%)", fontsize=7.5, labelpad=4)
    cbar.ax.tick_params(labelsize=7)
 
    # fig.text(
    #     0.44, 0.01,
    #     "× = degenerate (recall < 0.05)     — = baseline EOpp < 0.05",
    #     ha="center", fontsize=6.5, color="#666666",
    # )
    save(fig, "x1_crossdomain_heatmap")
 



"""
figure 12: Cross-Domain Bar Chart — Avg EOpp Reduction per Method

Each bar = average % EOpp reduction across 3 models × 3 sensitive attrs,
for one mitigation method in one domain.
Error bars = std across model×attribute combos (shows variability).
Positive bar = method reduced bias on average. Negative = made it worse.
Degenerate cases (recall < 0.05) excluded before averaging.

★ NHANES (blue): all four methods positive (+34% to +54%).
  EG best (+54%), RW close behind (+49%), TO and Supp lower but still solid.
  All error bars moderate — methods work fairly consistently.

★ OULAD (orange): only EG positive (+23%), everything else near-zero or negative.
  TO = -77%, driven by RF+TO catastrophic backfire (-236%).
  RW = -5%, Supp = +2% — basically no effect or slight harm.

Key takeaway:
1. EG is the only method that transfers across domains.
2. NHANES has deep bias → methods have room to improve.
   OULAD has mild bias → methods overshoot or have nothing to fix.
3. TO is the riskiest method: best-case +43% (NHANES), worst-case -77% (OULAD).
   High variance = not safe to deploy without domain validation.
4. RW and Supp are near-zero in OULAD — harmless but useless.
"""

def x2_crossdomain_bar():
    ieee_style()
    methods = ["Reweighing", "ExponentiatedGradient", "ThresholdOptimizer", "Suppression"]
 
    def _per_model_reduction(df, attrs, recall_thr=0.05, base_thr=0.05):
        out = {}
        for method in methods:
            model_reds = []
            for model in MODEL_ORDER:
                attr_reds = []
                for attr in attrs:
                    base_row = df[
                        (df["model"] == model)
                        & (df["sensitive_attr"] == attr)
                        & (df["miti_method"] == "Baseline")
                    ]
                    meth_row = df[
                        (df["model"] == model)
                        & (df["sensitive_attr"] == attr)
                        & (df["miti_method"] == method)
                    ]
                    if len(base_row) == 0 or len(meth_row) == 0:
                        continue
                    base_eopp = base_row["Equal Opportunity"].values[0]
                    meth_eopp = meth_row["Equal Opportunity"].values[0]
                    meth_recall = meth_row["recall"].values[0]
                    if meth_recall < recall_thr or base_eopp < base_thr:
                        continue
                    attr_reds.append((base_eopp - meth_eopp) / base_eopp * 100)
                if attr_reds:
                    model_reds.append(np.mean(attr_reds))
            out[method] = model_reds
        return out
 
    nh = _per_model_reduction(mit_df, ATTR)
    ou = _per_model_reduction(oulad_df, OULAD_ATTR)
 
    fig, ax = plt.subplots(figsize=(7.16, 3.0))
    x = np.arange(len(methods))
    w = 0.35
 
    for offset, data, label, color in [
        (-w / 2, nh, "NHANES (Healthcare)", "#2980b9"),
        (w / 2, ou, "OULAD (Education)", "#e67e22"),
    ]:
        means = [np.mean(data[m]) if data[m] else 0 for m in methods]
        lo = [np.mean(data[m]) - min(data[m]) if len(data[m]) > 1 else 0 for m in methods]
        hi = [max(data[m]) - np.mean(data[m]) if len(data[m]) > 1 else 0 for m in methods]
 
        ax.bar(
            x + offset, means, w, label=label, color=color,
            edgecolor="white", linewidth=0.5,
        )
        ax.errorbar(
            x + offset, means, yerr=[lo, hi], fmt="none",
            ecolor="#333333", capsize=3, capthick=0.8, linewidth=0.8,
        )
        for i, v in enumerate(means):
            if v != 0:
                sign = "+" if v > 0 else ""
                vy = v + (3 if v >= 0 else -3)
                va = "bottom" if v >= 0 else "top"
                ax.text(
                    x[i] + offset, vy, f"{sign}{v:.0f}%",
                    ha="center", va=va, fontsize=7,
                )
 
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_SHORT[m] for m in methods])
    ax.set_ylabel("Avg EOpp reduction (%)")
    ax.set_title("Cross-domain mitigation effectiveness")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)
 
    # clip y so the RF+TO outlier does not dominate
    y_lo = max(
        min(min(np.mean(d[m]) for m in methods if d[m]) for d in [nh, ou]) - 15,
        -90,
    )
    ax.set_ylim(y_lo, 85)
 
    # annotate clipped bar
    ou_to_mean = np.mean(ou["ThresholdOptimizer"]) if ou["ThresholdOptimizer"] else 0
    if ou_to_mean < y_lo:
        ax.annotate(
            f"{ou_to_mean:.0f}%\n(RF backfire)",
            xy=(x[2] + w / 2, y_lo + 2),
            ha="center", va="bottom", fontsize=6.5,
            color="#c0392b", fontweight="bold",
        )
        ax.annotate(
            "", xy=(x[2] + w / 2, y_lo),
            xytext=(x[2] + w / 2, y_lo + 8),
            arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.2),
        )
 
    # ax.text(
    #     0.5, -0.12,
    #     "Error bars = range across 3 models. "
    #     "Excludes degenerate (recall < 0.05) and near-zero baselines (EOpp < 0.05)",
    #     transform=ax.transAxes, ha="center", fontsize=6, color="#666666",
    # )
    fig.tight_layout()
    save(fig, "x2_crossdomain_bar")
 
 




if __name__ == "__main__":
    r1_baseline_performance()
    f1_baseline_details_dumbbell()
    r2_baseline_detais_tpr_fpr()
    r3_fairness_mit_grid_full()
    r4_performance_mit_grid_full()
    r5_disparity_reduction()
    r6_disparity_reduction()
    f2_tradeoff_f1_EOpp()
    f3_method_gap_comparison()
    f4_mit_detail_gap_dumbbell()
    x1_crossdomain_heatmap()
    x2_crossdomain_bar()
    