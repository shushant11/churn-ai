"""All report figures. Matplotlib only, saved as PNG into reports/figures/."""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve
from . import config

C = ["#2a6fdb", "#e3822b", "#2e9d6a", "#b8457e", "#7a5cc7", "#8a8a8a"]
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e6e6e6", "grid.linewidth": 0.6, "font.size": 9, "lines.linewidth": 2})


def _save(fig, name):
    p = config.FIGURES / name
    fig.tight_layout(); fig.savefig(p); plt.close(fig)
    return p


def pr_roc(curves: dict, y, base_rate):
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.8))
    for i, (name, p) in enumerate(curves.items()):
        pr, rc, _ = precision_recall_curve(y, p); ax[0].plot(rc, pr, color=C[i % 6], label=name)
        fpr, tpr, _ = roc_curve(y, p); ax[1].plot(fpr, tpr, color=C[i % 6], label=name)
    ax[0].axhline(base_rate, color="#999", ls="--", lw=1, label=f"no-skill ({base_rate:.2f})")
    ax[1].plot([0, 1], [0, 1], color="#999", ls="--", lw=1)
    ax[0].set(title="Precision-Recall (test)", xlabel="Recall", ylabel="Precision")
    ax[1].set(title="ROC (test)", xlabel="False positive rate", ylabel="True positive rate")
    ax[0].legend(frameon=False, fontsize=8)
    return _save(fig, "pr_roc.png")


def calibration(y, raw, cal):
    fig, ax = plt.subplots(figsize=(4.4, 4))
    for i, (lab, p) in enumerate([("raw", raw), ("calibrated", cal)]):
        f, m = calibration_curve(y, p, n_bins=10, strategy="quantile")
        ax.plot(m, f, marker="o", ms=4, color=C[i], label=lab)
    ax.plot([0, 1], [0, 1], color="#999", ls="--", lw=1, label="perfect")
    ax.set(title="Calibration (test)", xlabel="Predicted probability", ylabel="Observed churn rate"); ax.legend(frameon=False)
    return _save(fig, "calibration.png")


def gain(dec):
    fig, ax = plt.subplots(figsize=(4.6, 4))
    x = np.r_[0, dec["decile"] * 10]; ycap = np.r_[0, dec["cum_capture"] * 100]
    ax.plot(x, ycap, marker="o", ms=4, color=C[0], label="model"); ax.plot([0, 100], [0, 100], color="#999", ls="--", lw=1, label="random")
    ax.set(title="Cumulative gain (test)", xlabel="% customers contacted (riskiest first)", ylabel="% churners caught"); ax.legend(frameon=False)
    return _save(fig, "gain.png")


def profit(curve, t_star):
    fig, ax = plt.subplots(figsize=(5, 3.6))
    ax.plot(curve["threshold"], curve["profit"], color=C[0])
    ax.axvline(t_star, color=C[1], lw=1.2, ls="--", label=f"chosen on validation = {t_star:.2f}")
    ax.set(title="Expected campaign profit vs threshold (test)", xlabel="Threshold", ylabel="Profit (assumed units)"); ax.legend(frameon=False)
    return _save(fig, "profit.png")


def importance(imp):
    d = imp.iloc[::-1]
    fig, ax = plt.subplots(figsize=(5, 3.8))
    ax.barh(d["feature"], d["importance"], xerr=d["std"], color=C[0], height=0.6, error_kw={"lw": 0.8})
    ax.set(title="Permutation importance (drop in PR-AUC)", xlabel="PR-AUC drop when shuffled")
    return _save(fig, "importance.png")


def km(curves: dict, title):
    fig, ax = plt.subplots(figsize=(5, 3.8))
    for i, (g, d) in enumerate(curves.items()):
        ax.step(d["tenure"], d["survival"], where="post", color=C[i % 6], label=g)
    ax.set(title=title, xlabel="Tenure (years)", ylabel="Share not yet churned", ylim=(0.3, 1.01)); ax.legend(frameon=False)
    return _save(fig, "km_" + title.split()[-1].lower() + ".png")


def model_compare(rows):
    names = [r["model"] for r in rows]; v = [r["val_pr_auc"] for r in rows]
    fig, ax = plt.subplots(figsize=(5.2, 0.42 * len(names) + 1))
    ax.barh(names[::-1], v[::-1], color=C[0], height=0.6)
    for i, x in enumerate(v[::-1]):
        ax.text(x + 0.003, i, f"{x:.3f}", va="center", fontsize=8)
    ax.set(title="Validation PR-AUC by model", xlim=(min(v) - 0.05, max(v) + 0.04))
    return _save(fig, "model_compare.png")
