import numpy as np
from churnlab import config
from churnlab.business import best_threshold, decile_table, profit_curve, top_k_capture
from churnlab.evaluate import PlattCalibrator, at_threshold, bootstrap_ci, ece

rng = np.random.default_rng(0)
y = rng.binomial(1, 0.2, 4000)
p = np.clip(0.2 + 0.5 * (y - 0.2) + rng.normal(0, 0.15, 4000), 0.001, 0.999)


def test_profit_formula():
    assert config.BUSINESS.profit(tp=10, fp=10) == 10 * (0.3 * 500 - 50) - 10 * 50


def test_threshold_and_curve():
    t = best_threshold(y, p)
    c = profit_curve(y, p)
    assert 0 < t < 1 and c.profit.max() == c.loc[c.threshold.sub(t).abs().idxmin(), "profit"]


def test_deciles_and_capture():
    d = decile_table(y, p)
    assert len(d) == 10 and d.lift.iloc[0] > 1.5 and abs(d.cum_capture.iloc[-1] - 1) < 1e-9
    assert top_k_capture(y, p, 0.2) > 0.2


def test_calibration_reduces_ece():
    raw = np.clip(p * 1.6, 0, 0.999)                 # deliberately over-confident
    cal = PlattCalibrator().fit(raw[:2000], y[:2000]).transform(raw[2000:])
    assert ece(y[2000:], cal) < ece(y[2000:], raw[2000:])


def test_at_threshold_counts_and_ci():
    r = at_threshold(y, p, 0.5)
    assert r["tp"] + r["fp"] + r["fn"] + r["tn"] == len(y)
    lo, hi = bootstrap_ci(y, p, lambda a, b: float(np.mean((b >= 0.5) == a)), n=200)
    assert lo < hi
