"""
Shubin Li

whole preprocessing

data clean | feature engineering | 
"""

import pandas as pd
import numpy as np

# fix xpt zeros ->5.397605e-79
def _fix_xpt_zeros(df:pd.DataFrame) -> pd.DataFrame:
    epsilon=1e-10
    total_fixed=0
    num_cols= df.select_dtypes(include='number').columns
    for col in num_cols:
        mask= df[col].abs()<epsilon
        total_fixed += mask.sum()
        df.loc[mask,col]=0
    
    if total_fixed>0:
        print(f" _fix_xpt_zeros:{total_fixed} be fixed")
    return df


def _check_missing(df:pd.DataFrame):
    total_num=len(df)
    for c in df.columns:
        num=df[c].isna().sum()
        if num>0:
            print(f"{c} missing {num}, missing percent: {num/total_num :.1%}")
    print("check missing over")


# demo
def process_demo(df:pd.DataFrame) -> pd.DataFrame:
    # filter: age>18 sync with depression questionnaire
    # build col : income_group - sensitive variable
    # deal with refuse and don't know category -> NaN

    # RIDRETH3： 1	Mexican American 2	Other Hispanic 3	Non-Hispanic White  4	Non-Hispanic Black  6	Non-Hispanic Asian 7	Other Race - Including Multi-Racial
    # combine mexican american and other hispanic - > Hispanic
    
    demo=df.copy()

    demo=_fix_xpt_zeros(demo)

    keep_col=[
        'SEQN',     # Respondent sequence number
        'RIAGENDR', # Gender - sensitive variable
        'RIDAGEYR', # Age in years at screening
        'RIDRETH3', # Race/Hispanic origin w/ NH Asian - sensitive variable
        'DMDEDUC2', # Education level - Adults 20+
        'DMDMARTZ', # Marital status
        'INDFMPIR' # Ratio of family income to poverty
    ]

    #keep col
    demo=demo[keep_col]
    
    #filter
    demo=demo[demo['RIDAGEYR']>=18].copy()
    
    #build col  
    # income_group  dtype category ordered
    demo['income_group']=pd.cut(demo['INDFMPIR'],bins=[-np.inf,1.3,1.85,np.inf],labels=['Low income', 'Near poverty', 'Above threshold'])

    race_map = {1: "Hispanic", 2: "Hispanic",
                  3: "White", 4: "Black",
                  6: "Asian", 7: "Other Race"}
    # race_group dtype category unordered
    demo['race_group']=demo['RIDRETH3'].map(race_map)
    demo['race_group']=demo['race_group'].astype('category')

    #DMDMARTZ - Marital status  77	Refused  99	Don't know  -> NaN
    #DMDEDUC2 - Education level - Adults 20+ 7	Refused 9	Don't know
    demo['DMDMARTZ']=demo['DMDMARTZ'].replace([77,99],np.nan)
    demo['DMDEDUC2']=demo['DMDEDUC2'].replace([7,9],np.nan)

    _check_missing(demo)
    print(f' process_demo success: shape {demo.shape}')
    return demo


def process_dpq(df:pd.DataFrame) -> pd.DataFrame:

    dpq=df.copy()

    dpq=_fix_xpt_zeros(dpq)

    # only SEQN and PHQ-9 score and target(depressed) needed, build PHQ-9 and depressed (target)
    # deal with refuse and don't know category -> NaN
    
    #DPQ010 - DPQ090
    phq_9items=[f'DPQ0{i}0' for i in range(1,10)]
    #7	Refused 9	Don't know 
    for col in phq_9items:
        dpq[col]=dpq[col].replace([7,9],np.nan)

    # build PHQ9_score only include complete all 9 questions
    dpq['PHQ9_score']=dpq[phq_9items].sum(axis=1,min_count=9)
    
    dpq=dpq.dropna(subset=['PHQ9_score'])
    
    #build target  1 -> depressed (positive class) 2-> not depressed
    dpq["depressed"] = (dpq["PHQ9_score"] >= 10).astype(int)

    dpq=dpq[['SEQN','PHQ9_score','depressed']].copy()
    _check_missing(dpq)
    print(f' process_demo success: shape {dpq.shape}')
    return dpq



def process_slq(df:pd.DataFrame) -> pd.DataFrame:

    slq=df.copy()

    slq=_fix_xpt_zeros(slq)

    #SLD012 - Sleep hours - weekdays or workdays
    #SLD013 - Sleep hours - weekends
    # build a sleep_catchup  SLD013-SLD012 
    slq['sleep_catchup']=slq['SLD013']-slq['SLD012']

    slq=slq[['SEQN','SLD012','sleep_catchup']]
    
    _check_missing(slq)
    print(f' process_slq success: shape {slq.shape}')
    return slq



# 中等强度和高强度的 次数、频率 和时间分钟数。 需要三个合并在一块 统一时间段 算出 运动总时间
# PAD680 久坐时间 分钟

# Physical Activity
def process_paq(df:pd.DataFrame) -> pd.DataFrame:

    paq=df.copy()

    paq=_fix_xpt_zeros(paq)

    #PAD680 - Minutes sedentary activity
    # PAD790Q x PAD790U x PAD800    +   PAD810Q x PAD810U x PAD820 
    # PAD790Q\PAD800\PAD810Q\PAD820\PAD680 7777,9999 -> nan
    deal_cols=['PAD790Q','PAD800','PAD810Q','PAD820','PAD680']
    for col in deal_cols:
        paq[col]=paq[col].replace([7777,9999],np.nan)
    
    unit_to_week = {'D': 7, 'W': 1, 'M': 1/4.345, 'Y': 1/52}

    mod_min_wk=(paq['PAD790U'].map(unit_to_week)) * paq['PAD790Q'] * paq['PAD800']
    vig_min_wk=(paq['PAD810U'].map(unit_to_week)) * paq['PAD810Q'] * paq['PAD820']

    # if the Frequency is 0, min_wk should be 0 not Nan
    mod_min_wk=mod_min_wk.where(paq['PAD790Q']!=0,0)
    vig_min_wk=vig_min_wk.where(paq['PAD810Q']!=0,0)
    # MET = Metabolic Equivalent of Task assume vig_pa = 2 * mod_pa 
    paq['total_pa_min_wk']= mod_min_wk + vig_min_wk * 2

    paq=paq[['SEQN','PAD680','total_pa_min_wk']]

    _check_missing(paq)
    print(f' process_paq success: shape {paq.shape}')
    return paq


