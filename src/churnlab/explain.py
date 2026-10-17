"""Why did the model say that?  Global view = permutation importance (or SHAP if installed).
Local view (one customer) = 'what if this field were typical?' -> change in churn probability."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from .features import RAW


def global_importance(model, X: pd.DataFrame, y, seed: int = 42, n_repeats: int = 10) -> pd.DataFrame:
    """Shuffle one raw column at a time and measure how much PR-AUC drops. Big drop = the model relies on it."""
    r = permutation_importance(model, X[RAW], y, scoring="average_precision", n_repeats=n_repeats, random_state=seed, n_jobs=-1)
    return (pd.DataFrame({"feature": RAW, "importance": r.importances_mean, "std": r.importances_std})
            .sort_values("importance", ascending=False).reset_index(drop=True))


def _py(v):
    """numpy scalars -> plain Python, so results can be turned into JSON (FastAPI, Streamlit, MCP)."""
    return v.item() if hasattr(v, "item") else v


def typical_values(background: pd.DataFrame) -> dict:
    out = {}
    for c in RAW:
        s = background[c]
        out[c] = _py(s.mode().iloc[0]) if not pd.api.types.is_numeric_dtype(s) or s.nunique() <= 4 else float(s.median())
    return out


def local_drivers(predict_fn, row: pd.DataFrame, background: pd.DataFrame, top: int = 5, skip: tuple = ()) -> list[dict]:
    """For each field: replace the customer's value with the typical value and re-score.
    contribution = p(actual) - p(typical). Positive = this field pushes the customer TOWARDS churn."""
    base = float(predict_fn(row)[0])
    typ = typical_values(background)
    rows = []
    for c in [c for c in RAW if c not in skip]:
        alt = row.copy()
        alt[c] = typ[c]
        rows.append({"feature": c, "value": row[c].iloc[0], "typical": typ[c], "contribution": base - float(predict_fn(alt)[0])})
    rows.sort(key=lambda d: abs(d["contribution"]), reverse=True)
    for d in rows:
        d["value"], d["typical"] = _py(d["value"]), _py(d["typical"])
        d["contribution"] = round(float(d["contribution"]), 4)
    return rows[:top]


def shap_values(model, X: pd.DataFrame, background: pd.DataFrame):
    """Optional: exact-ish Shapley values with the `shap` library (run on your laptop: pip install shap)."""
    import shap  # noqa: raises ImportError if not installed
    f = lambda d: model.predict_proba(pd.DataFrame(d, columns=RAW))[:, 1]
    explainer = shap.Explainer(f, background[RAW].sample(min(200, len(background)), random_state=0))
    return explainer(X[RAW])
