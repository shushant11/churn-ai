import numpy as np
import pandas as pd
from churnlab.data import load, xy
from churnlab.fairness import audit
from churnlab.monitoring import drift_report, psi, simulate_drift
from churnlab.survival import kaplan_meier, logrank


def test_psi_zero_for_same_and_high_for_shift():
    a = np.random.default_rng(0).normal(0, 1, 5000)
    assert psi(a, a) < 0.01 and psi(a, a + 1) > 0.25


def test_drift_detects_simulation():
    X, _ = xy(load())
    rep = drift_report(X.sample(3000, random_state=0), simulate_drift(X.sample(2000, random_state=1))).set_index("feature")
    assert rep.loc["Age", "status"] == "retrain" and rep.loc["CreditScore", "status"] == "stable"


def test_kaplan_meier_hand_example():
    km = kaplan_meier([1, 2, 2, 3], [1, 1, 0, 0])      # S(1)=3/4, S(2)=3/4*(1-1/3)=0.5
    assert km.set_index("tenure").loc[2, "survival"] == 0.5


def test_logrank_geography_significant():
    assert logrank(load(), "Geography")["p_value"] < 1e-6


def test_fairness_shapes():
    df = pd.DataFrame({"Gender": ["Male", "Female"] * 50, "Geography": ["France"] * 100})
    y = np.array([0, 1] * 50); p = np.linspace(0, 1, 100)
    r = audit(df, y, p, 0.5)
    assert set(r) == {"Gender", "Geography"} and "selection_rate_ratio" in r["Gender"]["disparity"]
