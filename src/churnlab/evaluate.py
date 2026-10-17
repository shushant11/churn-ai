"""Metrics that matter for an imbalanced problem: PR-AUC, recall/precision at a threshold, calibration (Brier, ECE),
and bootstrap confidence intervals so every number comes with an honest error bar."""
from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from . import config


class PlattCalibrator:
    """Learns p_calibrated = sigmoid(a * logit(p_raw) + b) on the VALIDATION set.
    Why: class weights push raw scores up, so a raw 0.7 is not a 70% chance. After calibration it is (close to) one."""

    def fit(self, p_raw, y):
        self.lr_ = LogisticRegression(C=1e6, max_iter=1000).fit(self._logit(p_raw), y)
        return self

    def transform(self, p_raw):
        return self.lr_.predict_proba(self._logit(p_raw))[:, 1]

    @staticmethod
    def _logit(p):
        p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p)).reshape(-1, 1)


def ece(y, p, bins: int = 10) -> float:
    """Expected Calibration Error: average gap between predicted probability and observed churn rate, per bin."""
    y, p = np.asarray(y), np.asarray(p)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(total)


def at_threshold(y, p, t: float) -> dict:
    yhat = (np.asarray(p) >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    return {"threshold": round(float(t), 4), "precision": float(precision_score(y, yhat, zero_division=0)),
            "recall": float(recall_score(y, yhat)), "f1": float(f1_score(y, yhat)), "flagged_share": float(yhat.mean()),
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn), "profit": float(config.BUSINESS.profit(tp, fp))}


def ranking(y, p) -> dict:
    return {"roc_auc": float(roc_auc_score(y, p)), "pr_auc": float(average_precision_score(y, p)),
            "brier": float(brier_score_loss(y, p)), "ece": ece(y, p), "base_rate": float(np.mean(y))}


def bootstrap_ci(y, p, metric, n: int = 1000, seed: int = config.SEED, alpha: float = 0.05):
    """Resample the test set with replacement n times -> 95% interval of the metric."""
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        vals.append(metric(y[i], p[i]))
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def f1_threshold(y, p) -> float:
    grid = np.linspace(0.05, 0.95, 181)
    return float(grid[int(np.argmax([f1_score(y, (p >= t).astype(int)) for t in grid]))])
