"""Fairness audit: does the model treat groups (Gender, Geography) very differently?
We REPORT gaps; we do not silently 'fix' them, because some gaps reflect real differences in churn (Germany churns ~2x)."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def group_report(df: pd.DataFrame, y, p, threshold: float, attr: str) -> pd.DataFrame:
    y, p = np.asarray(y), np.asarray(p)
    flag = p >= threshold
    rows = []
    for g in sorted(df[attr].unique()):
        m = (df[attr] == g).values
        yy, ff = y[m], flag[m]
        rows.append({"attribute": attr, "group": g, "n": int(m.sum()), "actual_churn_rate": float(yy.mean()),
                     "selection_rate": float(ff.mean()),                                    # demographic parity
                     "recall": float(ff[yy == 1].mean()) if (yy == 1).any() else np.nan,   # equal opportunity
                     "fpr": float(ff[yy == 0].mean()) if (yy == 0).any() else np.nan,      # equalised odds (other half)
                     "auc": float(roc_auc_score(yy, p[m])) if 0 < yy.mean() < 1 else np.nan})
    return pd.DataFrame(rows)


def disparity(rep: pd.DataFrame) -> dict:
    """min/max ratio per metric. The '80% rule' of thumb flags a ratio below 0.8."""
    out = {}
    for col in ["selection_rate", "recall", "fpr"]:
        v = rep[col].dropna()
        out[f"{col}_ratio"] = float(v.min() / v.max()) if v.max() > 0 else np.nan
    out["flag_80pct_rule"] = bool(out["selection_rate_ratio"] < 0.8)
    return out


def audit(df, y, p, threshold, attrs=("Gender", "Geography")) -> dict:
    res = {}
    for a in attrs:
        rep = group_report(df, y, p, threshold, a)
        res[a] = {"groups": rep.round(4).to_dict(orient="records"), "disparity": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in disparity(rep).items()}}
    return res
