"""Feature engineering as a scikit-learn pipeline step, so training and serving can never drift apart."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from . import config
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

RAW = ["CreditScore", "Geography", "Gender", "Age", "Tenure", "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"]
NUMERIC = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "EstimatedSalary", "BalanceSalaryRatio"]
BINARY = ["HasCrCard", "IsActiveMember", "ZeroBalance", "MultiProduct", "Male"]
CATEGORICAL = ["Geography"]


def add_features(df: pd.DataFrame, drop_gender: bool = False) -> pd.DataFrame:
    d = df[RAW].copy()
    d["BalanceSalaryRatio"] = d["Balance"] / (d["EstimatedSalary"] + 1.0)
    d["ZeroBalance"] = (d["Balance"] == 0).astype(int)
    d["MultiProduct"] = (d["NumOfProducts"] >= 3).astype(int)     # 3-4 products churn at ~83-100% in this data
    if drop_gender:                                  # gender-blind model: the column is not even created
        return d.drop(columns=["Gender"])
    d["Male"] = (d["Gender"] == "Male").astype(int)
    return d


class AddFeatures:
    """Picklable feature builder that remembers whether the model is gender-blind (so serving can never disagree with training)."""
    def __init__(self, drop_gender: bool = True):
        self.drop_gender = drop_gender

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        return add_features(df, self.drop_gender)


def make_preprocessor(scale: bool = True, drop_gender: bool | None = None) -> Pipeline:
    drop_gender = config.DROP_GENDER if drop_gender is None else drop_gender
    ct = ColumnTransformer([
        ("num", StandardScaler() if scale else "passthrough", NUMERIC),
        ("bin", "passthrough", [c for c in BINARY if not (drop_gender and c == "Male")]),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
    ], verbose_feature_names_out=False)
    ct.set_output(transform="pandas")
    return Pipeline([("features", FunctionTransformer(AddFeatures(drop_gender), validate=False)), ("encode", ct)])
