"""
Shubin Li

train : LR , RF , XGBoost

train_test split


"""

import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

RANDOM_STATE = 42
TEST_SIZE = 0.2


def _train_test_split(df: pd.DataFrame):
    X = df.drop(columns=["depressed"])
    y = df["depressed"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    return X_train, X_test, y_train, y_test

def get3_pipe_models()->dict[str,Pipeline]:
    """
    class imbalance  ~13% depressed, handle it with 
    LR/RF  : class_weight="balanced"
    XGBoost: scale_pos_weight = n_neg / n_pos
    """ 
