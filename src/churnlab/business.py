"""Turns probabilities into decisions a bank manager understands: who to call, how many, and what it is worth."""
from __future__ import annotations
import numpy as np
import pandas as pd
from . import config


def profit_curve(y, p, grid=None) -> pd.DataFrame:
    """Expected campaign profit for every threshold. Profit = TP*(success*value - cost) - FP*cost (see config.Business)."""
    y, p = np.asarray(y), np.asarray(p)
    grid = np.linspace(0.02, 0.98, 97) if grid is None else grid
    rows = []
    for t in grid:
        flag = p >= t
        tp, fp = int((flag & (y == 1)).sum()), int((flag & (y == 0)).sum())
        rows.append({"threshold": float(t), "contacted": int(flag.sum()), "tp": tp, "fp": fp, "profit": config.BUSINESS.profit(tp, fp)})
    return pd.DataFrame(rows)


def best_threshold(y, p) -> float:
    """Choose the threshold on VALIDATION data only; the test set just reports what that choice earns."""
    c = profit_curve(y, p)
    return float(c.loc[c["profit"].idxmax(), "threshold"])


def decile_table(y, p, n: int = 10) -> pd.DataFrame:
    """Sort customers by risk, cut into 10 equal groups. Lift = churn rate in the group / overall churn rate."""
    d = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)}).sort_values("p", ascending=False).reset_index(drop=True)
    d["decile"] = np.arange(len(d)) * n // len(d) + 1
    g = d.groupby("decile").agg(customers=("y", "size"), churners=("y", "sum"), avg_score=("p", "mean"))
    g["churn_rate"] = g["churners"] / g["customers"]
    g["lift"] = g["churn_rate"] / d["y"].mean()
    g["cum_capture"] = g["churners"].cumsum() / d["y"].sum()      # the "gain" curve
    return g.reset_index()


def top_k_capture(y, p, share: float = 0.2) -> float:
    """Share of all churners caught if the bank can only call the riskiest `share` of customers."""
    y, p = np.asarray(y), np.asarray(p)
    k = int(round(len(y) * share))
    return float(y[np.argsort(-p)[:k]].sum() / y.sum())
