"""
Shubin Li

whole preprocessing

data clean | feature engineering | merge
"""

import pandas as pd
import numpy as np


# fix xpt zeros ->5.397605e-79
def _fix_xpt_zeros(df: pd.DataFrame) -> pd.DataFrame:
    epsilon = 1e-10
    total_fixed = 0
    num_cols = df.select_dtypes(include="number").columns
    for col in num_cols:
        mask = df[col].abs() < epsilon
        total_fixed += mask.sum()
        df.loc[mask, col] = 0

    if total_fixed > 0:
        print(f" _fix_xpt_zeros:{total_fixed} be fixed")
    return df


def _check_missing(df: pd.DataFrame):
    total_num = len(df)
    for c in df.columns:
        num = df[c].isna().sum()
        if num > 0:
            print(f"{c} missing {num}, missing percent: {num/total_num :.1%}")
    print("check missing over")


# Demographic Variables (DEMO_L)
def process_demo(df: pd.DataFrame) -> pd.DataFrame:
    # filter: age>18 sync with depression questionnaire
    # build col : income_group - sensitive variable
    # deal with refuse and don't know category -> NaN

    # RIDRETH3： 1	Mexican American 2	Other Hispanic 3	Non-Hispanic White  4	Non-Hispanic Black  6	Non-Hispanic Asian 7	Other Race - Including Multi-Racial
    # combine mexican american and other hispanic - > Hispanic

    demo = df.copy()

    demo = _fix_xpt_zeros(demo)

    keep_col = [
        "SEQN",  # Respondent sequence number
        "RIAGENDR",  # Gender - sensitive variable
        "RIDAGEYR",  # Age in years at screening
        "RIDRETH3",  # Race/Hispanic origin w/ NH Asian - sensitive variable
        "DMDEDUC2",  # Education level - Adults 20+
        "DMDMARTZ",  # Marital status
        "INDFMPIR",  # Ratio of family income to poverty
    ]

    # keep col
    demo = demo[keep_col]

    # filter
    demo = demo[demo["RIDAGEYR"] >= 18].copy()

    # build col
    # income_group  dtype category ordered
    demo["income_group"] = pd.cut(
        demo["INDFMPIR"],
        bins=[-np.inf, 1.3, 1.85, np.inf],
        labels=["Low income", "Near poverty", "Above threshold"],
    )

    race_map = {
        1: "Hispanic",
        2: "Hispanic",
        3: "White",
        4: "Black",
        6: "Asian",
        7: "Other Race",
    }
    # race_group dtype category unordered
    demo["race_group"] = demo["RIDRETH3"].map(race_map)
    demo["race_group"] = demo["race_group"].astype("category")

    # DMDMARTZ - Marital status  77	Refused  99	Don't know  -> NaN
    # DMDEDUC2 - Education level - Adults 20+ 7	Refused 9	Don't know
    demo["DMDMARTZ"] = demo["DMDMARTZ"].replace([77, 99], np.nan)
    demo["DMDEDUC2"] = demo["DMDEDUC2"].replace([7, 9], np.nan)

    _check_missing(demo)
    print(f" process_demo success: shape {demo.shape}\n")
    return demo


# Mental Health - Depression Screener (DPQ_L)
def process_dpq(df: pd.DataFrame) -> pd.DataFrame:

    dpq = df.copy()

    dpq = _fix_xpt_zeros(dpq)

    # only SEQN and PHQ-9 score and target(depressed) needed, build PHQ-9 and depressed (target)
    # deal with refuse and don't know category -> NaN

    # DPQ010 - DPQ090
    phq_9items = [f"DPQ0{i}0" for i in range(1, 10)]
    # 7	Refused 9	Don't know
    for col in phq_9items:
        dpq[col] = dpq[col].replace([7, 9], np.nan)

    # build PHQ9_score only include complete all 9 questions
    dpq["PHQ9_score"] = dpq[phq_9items].sum(axis=1, min_count=9)

    dpq = dpq.dropna(subset=["PHQ9_score"])

    # build target  1 -> depressed (positive class) 2-> not depressed
    dpq["depressed"] = (dpq["PHQ9_score"] >= 10).astype(int)

    dpq = dpq[["SEQN", "PHQ9_score", "depressed"]].copy()
    _check_missing(dpq)
    print(f" process_dpq success: shape {dpq.shape}\n")
    return dpq


# Sleep Disorders (SLQ_L)
def process_slq(df: pd.DataFrame) -> pd.DataFrame:

    slq = df.copy()

    slq = _fix_xpt_zeros(slq)

    # SLD012 - Sleep hours - weekdays or workdays
    # SLD013 - Sleep hours - weekends
    # build a sleep_catchup  SLD013-SLD012
    slq["sleep_catchup"] = slq["SLD013"] - slq["SLD012"]

    slq = slq[["SEQN", "SLD012", "sleep_catchup"]]

    _check_missing(slq)
    print(f" process_slq success: shape {slq.shape}\n")
    return slq


# 中等强度和高强度的 次数、频率 和时间分钟数。 需要三个合并在一块 统一时间段 算出 运动总时间
# PAD680 久坐时间 分钟


# Physical Activity (PAQ_L)
def process_paq(df: pd.DataFrame) -> pd.DataFrame:

    paq = df.copy()

    paq = _fix_xpt_zeros(paq)

    # PAD680 - Minutes sedentary activity
    # PAD790Q x PAD790U x PAD800    +   PAD810Q x PAD810U x PAD820
    # PAD790Q\PAD800\PAD810Q\PAD820\PAD680 7777,9999 -> nan
    deal_cols = ["PAD790Q", "PAD800", "PAD810Q", "PAD820", "PAD680"]
    for col in deal_cols:
        paq[col] = paq[col].replace([7777, 9999], np.nan)

    unit_to_week = {"D": 7, "W": 1, "M": 1 / 4.345, "Y": 1 / 52}

    mod_min_wk = (paq["PAD790U"].map(unit_to_week)) * paq["PAD790Q"] * paq["PAD800"]
    vig_min_wk = (paq["PAD810U"].map(unit_to_week)) * paq["PAD810Q"] * paq["PAD820"]

    # if the Frequency is 0, min_wk should be 0 not Nan
    mod_min_wk = mod_min_wk.where(paq["PAD790Q"] != 0, 0)
    vig_min_wk = vig_min_wk.where(paq["PAD810Q"] != 0, 0)
    # MET = Metabolic Equivalent of Task assume vig_pa = 2 * mod_pa
    paq["total_pa_min_wk"] = mod_min_wk + vig_min_wk * 2

    paq = paq[["SEQN", "PAD680", "total_pa_min_wk"]]

    _check_missing(paq)
    print(f" process_paq success: shape {paq.shape}\n")
    return paq


# Alcohol Use (ALQ_L)
def process_alq(df: pd.DataFrame) -> pd.DataFrame:
    """
    ALQ111 - Ever had a drink of any kind of alcohol
    ALQ121 - Past 12 mos how often drink alc bev   - need unify the frequency(alc_days_per_year)
    ALQ142 - # days have 4/5 drinks/past 12 mos   - need unify the frequency(binge_days_per_year)
    """
    alq = df.copy()
    alq = _fix_xpt_zeros(alq)

    # 2 categories  1 and 0
    alq["ever_drinker"] = alq["ALQ111"].replace({1: 1, 2: 0, 7: np.nan, 9: np.nan})

    alq["ALQ121"] = alq["ALQ121"].replace({77: np.nan, 99: np.nan})
    alq["ALQ142"] = alq["ALQ142"].replace({77: np.nan, 99: np.nan})

    # ever_drinker = 0 (2) -> ALQ121 = 0 ALQ142 = 0 not NaN
    alq.loc[alq["ever_drinker"] == 0, "ALQ121"] = 0
    alq.loc[alq["ever_drinker"] == 0, "ALQ142"] = 0

    # ALQ121 = 0  -> ALQ142 = 0 not NaN
    alq.loc[alq["ALQ121"] == 0, "ALQ142"] = 0

    freq_map = {
        0: 0,  # 从不
        1: 365,  # 每天
        2: 300,  # 几乎每天
        3: 182,  # 每周3-4次
        4: 104,  # 每周2次
        5: 52,  # 每周1次
        6: 30,  # 每月2-3次
        7: 12,  # 每月1次
        8: 9,  # 一年7-11次
        9: 4.5,  # 一年3-6次
        10: 1.5,  # 一年1-2次
    }
    alq["alc_days_per_year"] = alq["ALQ121"].map(freq_map)
    alq["binge_days_per_year"] = alq["ALQ142"].map(freq_map)

    alq = alq[["SEQN", "ever_drinker", "alc_days_per_year", "binge_days_per_year"]]
    _check_missing(alq)
    print(f" process_alq success: shape {alq.shape}\n")
    return alq


# Smoking - Cigarette Use (SMQ_L)
def process_smq(df: pd.DataFrame) -> pd.DataFrame:
    """
    SMQ020 - Smoked at least 100 cigarettes in life
    SMQ040 - Do you now smoke cigarettes?
    SMD650 - Avg # cigarettes/day during past 30 days
    """
    smq = df.copy()
    smq = _fix_xpt_zeros(smq)

    smq["SMQ020"] = smq["SMQ020"].replace({7: np.nan, 9: np.nan})
    smq["SMQ040"] = smq["SMQ040"].replace({7: np.nan, 9: np.nan})
    smq["SMD650"] = smq["SMD650"].replace({777: np.nan, 999: np.nan})

    # dealwith skip item
    smq["cigs_per_day"] = smq["SMD650"]
    smq.loc[smq["SMQ020"] == 2, "SMQ040"] = 3
    smq.loc[smq["SMQ020"] == 2, "cigs_per_day"] = 0
    smq.loc[smq["SMQ040"] == 3, "cigs_per_day"] = 0

    # combine SMQ020 and SMQ040  to a three categories (never smoke,former smoker,current smoke)
    conditions = [
        smq["SMQ020"] == 2,
        (smq["SMQ020"] == 1) & (smq["SMQ040"] == 3),
        (smq["SMQ020"] == 1) & (smq["SMQ040"].isin([1, 2])),
    ]
    choices = [0, 1, 2]

    smq["smoke_status"] = np.select(conditions, choices, default=np.nan)

    smq = smq[["SEQN", "smoke_status", "cigs_per_day"]]

    _check_missing(smq)
    print(f" process_smq success: shape {smq.shape}\n")
    return smq


# Income (INQ_L)
def process_inq(df: pd.DataFrame) -> pd.DataFrame:
    """
    INQ300 - Family has savings more than $20,000
    """
    inq = df.copy()
    inq = _fix_xpt_zeros(inq)

    inq["INQ300"] = inq["INQ300"].replace({7: np.nan, 9: np.nan})

    inq["has_saving"] = inq["INQ300"].map({1: 1, 2: 0})

    inq = inq[["SEQN", "has_saving"]]

    _check_missing(inq)
    print(f" process_inq success: shape {inq.shape}\n")
    return inq


# Health Insurance (HIQ_L)
def process_hiq(df: pd.DataFrame) -> pd.DataFrame:
    """
    HIQ011 - Covered by health insurance
    HIQ032D - Covered by Medicaid
    HIQ210 - Time when no insurance in past year?
    """
    hiq = df.copy()
    hiq = _fix_xpt_zeros(hiq)

    # has_insurance
    hiq["HIQ011"] = hiq["HIQ011"].replace({7: np.nan, 9: np.nan})
    hiq["has_insurance"] = hiq["HIQ011"].map({1: 1, 2: 0})

    # use_Medicaid HIQ011 2 no cover,  HIQ032D=4 cover, HIQ011=1 and HIQ032D!=4 no cover  others nan
    # it could be overlap with income
    # conditions = [hiq["HIQ032D"] == 4, hiq["HIQ011"].isin([1, 2])]
    # choices = [1, 0]
    # hiq["cover_by_Medicaid"] = np.select(conditions, choices, default=np.nan)

    # anytime no insurance last year
    # hiq["had_coverage_gap"] = hiq["HIQ210"].map({1: 1, 2: 0})

    hiq = hiq[["SEQN", "has_insurance"]]

    _check_missing(hiq)
    print(f" process_hiq success: shape {hiq.shape}\n")
    return hiq


# Diabetes (DIQ_L)
def process_diq(df: pd.DataFrame) -> pd.DataFrame:
    """
    DIQ010 - Doctor told you have diabetes
    """
    diq = df.copy()
    diq = _fix_xpt_zeros(diq)

    diq["DIQ010"] = diq["DIQ010"].replace({7: np.nan, 9: np.nan})

    # 2 categories  diabetes 1,Borderline 1,no diabetes 0
    diq["has_diabetes"] = diq["DIQ010"].map({1: 1, 2: 0, 3: 1})

    diq = diq[["SEQN", "has_diabetes"]]

    _check_missing(diq)
    print(f" process_diq success: shape {diq.shape}\n")
    return diq


# Body Measures (BMX_L)
def process_bmx(df: pd.DataFrame) -> pd.DataFrame:
    """
    BMXBMI - Body Mass Index (kg/m**2)  keep
    BMXWAIST - Waist Circumference (cm)
    BMXHT - Standing Height (cm)
    waist_to_height_ratio = BMXWAIST / BMXHT keep
    """
    bmx = df.copy()
    bmx = _fix_xpt_zeros(bmx)

    bmx["waist_to_height_ratio"] = bmx["BMXWAIST"] / bmx["BMXHT"]

    bmx = bmx[["SEQN", "BMXBMI", "waist_to_height_ratio"]]

    _check_missing(bmx)
    print(f" process_bmx success: shape {bmx.shape}\n")
    return bmx


_table_func_map = {
    "DEMO_L": process_demo,
    "DPQ_L": process_dpq,
    "SLQ_L": process_slq,
    "PAQ_L": process_paq,
    "ALQ_L": process_alq,
    "SMQ_L": process_smq,
    "INQ_L": process_inq,
    "HIQ_L": process_hiq,
    "DIQ_L": process_diq,
    "BMX_L": process_bmx,
}

_merge_order = [
    "DPQ_L",
    "DEMO_L",
    "SLQ_L",
    "PAQ_L",
    "ALQ_L",
    "SMQ_L",
    "INQ_L",
    "HIQ_L",
    "DIQ_L",
    "BMX_L",
]


# clean & feature engineering & merge
def processing_tables(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    new_tables = {}
    for t, func in _table_func_map.items():
        new_tables[t] = func(tables[t])

    merged = new_tables[_merge_order[0]]
    print(f"initial len:{len(merged)}")
    for t in _merge_order[1:]:
        merged = merged.merge(new_tables[t], on="SEQN", how="left")
        print(f"after merged len:{len(merged)}")

    print(f"merged over, shape: {merged.shape}\n")

    print("missing:" + "=" * 50)
    for col in merged.columns:
        missing_per = merged[col].isna().mean()
        print(f"{col} missing : {missing_per:.1%}")

    return merged


"""
Merged Dataset Fields (25 fields, 5455 records)
Primary table: DPQ_L, left-joined to 9 supplementary tables via SEQN.

SEQN                 Respondent ID, integer. From DPQ_L                                                                (drop before modeling)
PHQ9_score           PHQ-9 total score, 0–27 integer. From DPQ_L                                                       (drop before modeling)
depressed            Prediction target, 1 if PHQ9 >= 10, else 0. From DPQ_L                                            (target)
RIAGENDR             Gender, 1=Male 2=Female. From DEMO_L                                                              (sensitive attribute)
RIDAGEYR             Age at screening, 18+. From DEMO_L
RIDRETH3             Original race/ethnicity code, 1–7 (used to derive race_group). From DEMO_L                        (drop before modeling)
DMDEDUC2             Adult education level, 1–5 ordinal. From DEMO_L
DMDMARTZ             Marital status, 1=Married/Cohabiting 2=Widowed/Divorced/Separated 3=Never married. From DEMO_L
INDFMPIR             Family income-to-poverty ratio, 0–5 continuous (used to derive income_group). From DEMO_L
income_group         Income tier, Low income / Near poverty / Above threshold, 3-level ordinal. From DEMO_L             (sensitive attribute)
race_group           Race category, Hispanic / White / Black / Asian / Other Race, 5-level nominal. From DEMO_L         (sensitive attribute)
SLD012               Average weekday sleep hours, continuous. From SLQ_L
sleep_catchup        Weekend - weekday sleep difference, continuous, can be negative. From SLQ_L
PAD680               Daily sedentary time in minutes, continuous. From PAQ_L
total_pa_min_wk      Weekly total physical activity minutes, vigorous x2 weighted, continuous. From PAQ_L
ever_drinker         Lifetime alcohol use, 0/1. From ALQ_L
alc_days_per_year    Annualized drinking days, ordinal discrete (11 mapped values from 0–365). From ALQ_L
binge_days_per_year  Annualized binge drinking days, ordinal discrete (11 mapped values from 0–365). From ALQ_L
smoke_status         Smoking status, 0=Never 1=Former 2=Current, 3-level ordinal. From SMQ_L
cigs_per_day         Average daily cigarettes past 30 days, continuous. From SMQ_L
has_saving           Family savings > $20,000, 0/1. From INQ_L
has_insurance        Has health insurance, 0/1. From HIQ_L
has_diabetes         Diabetes or borderline, 0/1. From DIQ_L
BMXBMI               Body Mass Index (kg/m2), continuous. From BMX_L
waist_to_height_ratio Waist-to-height ratio (BMXWAIST / BMXHT), continuous. From BMX_L
"""


if __name__ == "__main__":
    import data_loader

    tables = data_loader.load_xpt_files()
    merged_table = processing_tables(tables)
    print("all table merged\n")
