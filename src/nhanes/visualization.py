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




if __name__ == "__main__":
    r1_baseline_performance()
    f1_baseline_details_dumbbell()
    r2_baseline_detais_tpr_fpr()
    r3_fairness_mit_grid_full()
    r4_performance_mit_grid_full()
    r5_disparity_reduction()
    r6_disparity_reduction()
    f2_tradeoff_f1_EOpp()