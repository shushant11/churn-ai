"""Survival view: not just 'will they leave' but 'how long do they stay'. Tenure (years with the bank) is the time axis,
Exited is the event. Kaplan-Meier written by hand so you can explain every line. (lifelines does the same if installed.)"""
from __future__ import annotations
import numpy as np
import pandas as pd


def kaplan_meier(durations, events) -> pd.DataFrame:
    """S(t) = product over event times t_i <= t of (1 - d_i / n_i)
    d_i = customers who churned at t_i, n_i = customers still 'at risk' (tenure >= t_i)."""
    t, e = np.asarray(durations), np.asarray(events)
    s, rows = 1.0, []
    for ti in np.sort(np.unique(t)):
        n_at_risk = int((t >= ti).sum())
        d = int(((t == ti) & (e == 1)).sum())
        s *= 1 - d / n_at_risk if n_at_risk else 1
        rows.append({"tenure": int(ti), "at_risk": n_at_risk, "churned": d, "survival": round(s, 4)})
    return pd.DataFrame(rows)


def by_group(df: pd.DataFrame, group: str, time="Tenure", event="Exited") -> dict:
    return {str(g): kaplan_meier(d[time], d[event]) for g, d in df.groupby(group)}


def logrank(df: pd.DataFrame, group: str, time="Tenure", event="Exited") -> dict:
    """Two-or-more-group log-rank test (chi-square). Small p = the survival curves really differ."""
    from scipy.stats import chi2
    groups = sorted(df[group].unique())
    times = np.sort(df.loc[df[event] == 1, time].unique())
    O = np.zeros(len(groups)); E = np.zeros(len(groups)); V = np.zeros((len(groups), len(groups)))
    for ti in times:
        at = df[df[time] >= ti]
        n = len(at); d = int(((at[time] == ti) & (at[event] == 1)).sum())
        if n < 2 or d == 0:
            continue
        ng = np.array([(at[group] == g).sum() for g in groups])
        dg = np.array([((at[group] == g) & (at[time] == ti) & (at[event] == 1)).sum() for g in groups])
        O += dg; E += d * ng / n
        frac = ng / n
        V += d * (n - d) / (n - 1) * (np.diag(frac) - np.outer(frac, frac))
    k = len(groups) - 1
    diff = (O - E)[:k]
    stat = float(diff @ np.linalg.pinv(V[:k, :k]) @ diff)
    return {"groups": groups, "observed": O.tolist(), "expected": [round(x, 1) for x in E], "chi2": round(stat, 2), "df": k, "p_value": float(chi2.sf(stat, k))}
