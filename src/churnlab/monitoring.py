"""Data drift: is today's customer data still like the data the model learned from?
PSI (Population Stability Index): <0.1 stable, 0.1-0.25 watch, >0.25 retrain. KS test for numeric shape changes."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from .features import RAW


def psi(ref, cur, bins: int = 10) -> float:
    ref, cur = pd.Series(ref), pd.Series(cur)
    if not pd.api.types.is_numeric_dtype(ref) or ref.nunique() <= 10:                       # categorical / small integer -> share per level
        levels = sorted(set(ref.unique()) | set(cur.unique()), key=str)
        r = ref.value_counts(normalize=True).reindex(levels, fill_value=0).values
        c = cur.value_counts(normalize=True).reindex(levels, fill_value=0).values
    else:                                                                 # numeric -> decile bins of the REFERENCE
        edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        r = np.histogram(ref, edges)[0] / len(ref)
        c = np.histogram(cur, edges)[0] / len(cur)
    r, c = np.clip(r, 1e-4, None), np.clip(c, 1e-4, None)
    return float(np.sum((c - r) * np.log(c / r)))


def drift_report(ref: pd.DataFrame, cur: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in RAW:
        v = psi(ref[col], cur[col])
        ks_p = float(ks_2samp(ref[col], cur[col]).pvalue) if pd.api.types.is_numeric_dtype(ref[col]) else np.nan
        rows.append({"feature": col, "psi": round(v, 4), "ks_p": ks_p, "status": "retrain" if v > 0.25 else "watch" if v > 0.1 else "stable"})
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


def simulate_drift(df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Demo only: a 'next quarter' where customers are older, balances fall 30% and more come from Germany."""
    rng = np.random.default_rng(seed)
    d = df.copy()
    d["Age"] = (d["Age"] + rng.normal(6, 3, len(d))).clip(18, 92).round().astype(int)
    d["Balance"] = d["Balance"] * 0.7
    flip = rng.random(len(d)) < 0.25
    d.loc[flip, "Geography"] = "Germany"
    return d
