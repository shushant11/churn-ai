"""Load + validate + split. Validation fails loudly instead of letting bad data train a model."""
from __future__ import annotations
import pandas as pd
from sklearn.model_selection import train_test_split
from . import config

EXPECTED = ["RowNumber", "CustomerId", "Surname", "CreditScore", "Geography", "Gender", "Age", "Tenure", "Balance",
            "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary", "Exited"]
RANGES = {"CreditScore": (300, 900), "Age": (18, 100), "Tenure": (0, 10), "Balance": (0, 1e7), "NumOfProducts": (1, 4),
          "HasCrCard": (0, 1), "IsActiveMember": (0, 1), "EstimatedSalary": (0, 1e7), "Exited": (0, 1)}
CATEGORIES = {"Geography": {"France", "Germany", "Spain"}, "Gender": {"Female", "Male"}}


class DataValidationError(ValueError):
    pass


def validate(df: pd.DataFrame) -> dict:
    problems = []
    missing_cols = [c for c in EXPECTED if c not in df.columns]
    if missing_cols:
        raise DataValidationError(f"missing columns: {missing_cols}")
    if df.isna().any().any():
        problems.append(f"missing values: {df.isna().sum()[df.isna().sum() > 0].to_dict()}")
    for c, (lo, hi) in RANGES.items():
        bad = int(((df[c] < lo) | (df[c] > hi)).sum())
        if bad:
            problems.append(f"{c}: {bad} values outside [{lo}, {hi}]")
    for c, ok in CATEGORIES.items():
        bad = set(df[c].unique()) - ok
        if bad:
            problems.append(f"{c}: unexpected categories {bad}")
    if df["CustomerId"].duplicated().any():
        problems.append("duplicate CustomerId rows")
    if problems:
        raise DataValidationError("; ".join(problems))
    return {"rows": len(df), "churn_rate": float(df[config.TARGET].mean()), "duplicates": 0, "missing": 0}


def load(path=config.DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    validate(df)
    return df


def split(df: pd.DataFrame, seed: int = config.SEED):
    """Stratified 60 / 20 / 20 train / validation / test. TEST is touched exactly once, at the very end."""
    y = df[config.TARGET]
    trval, te = train_test_split(df, test_size=0.2, stratify=y, random_state=seed)
    tr, va = train_test_split(trval, test_size=0.25, stratify=trval[config.TARGET], random_state=seed)   # 0.25 * 0.8 = 0.2
    return tr.reset_index(drop=True), va.reset_index(drop=True), te.reset_index(drop=True)


def xy(df: pd.DataFrame):
    return df.drop(columns=[config.TARGET] + [c for c in config.DROP_COLS if c in df.columns]), df[config.TARGET].values
