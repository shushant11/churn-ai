"""Blend + neural-model tests. Torch tests are skipped automatically when torch is not installed."""
import pickle
import numpy as np
import pytest
from sklearn.metrics import average_precision_score
from churnlab.data import load, split, xy
from churnlab.models import Blend, _pipe
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def _data():
    tr, va, _ = split(load())
    return (*xy(tr.head(1500)), *xy(va.head(500)))


def test_blend_is_mean_of_members_and_refits_after_clone():
    from sklearn.base import clone
    Xtr, ytr, Xva, yva = _data()
    a = _pipe(LogisticRegression(max_iter=500, class_weight="balanced")).fit(Xtr, ytr)
    b = _pipe(RandomForestClassifier(n_estimators=50, random_state=0), scale=False).fit(Xtr, ytr)
    bl = Blend.from_fitted(["lr", "rf"], [a, b])
    p = bl.predict_proba(Xva)[:, 1]
    assert np.allclose(p, (a.predict_proba(Xva)[:, 1] + b.predict_proba(Xva)[:, 1]) / 2)
    again = clone(bl).fit(Xtr, ytr)                                  # what the gender-blind ablation does
    assert average_precision_score(yva, again.predict_proba(Xva)[:, 1]) > 0.3
    assert pickle.loads(pickle.dumps(bl)).predict_proba(Xva).shape == (len(Xva), 2)


def test_torch_ft_transformer_trains_and_pickles():
    pytest.importorskip("torch")
    from churnlab.nn_models import TorchFT
    Xtr, ytr, Xva, yva = _data()
    m = _pipe(TorchFT(epochs=3, d_model=16, n_layers=1)).fit(Xtr, ytr)
    p = m.predict_proba(Xva)[:, 1]
    assert p.shape == (len(Xva),) and ((p >= 0) & (p <= 1)).all()
    assert np.allclose(pickle.loads(pickle.dumps(m)).predict_proba(Xva)[:, 1], p, atol=1e-5)


def test_torch_ann_accepts_tuning_params():
    pytest.importorskip("torch")
    from churnlab.nn_models import TorchANN
    Xtr, ytr, Xva, _ = _data()
    m = _pipe(TorchANN(hidden=(32,), dropout=0.1, weight_decay=1e-3, epochs=3)).fit(Xtr, ytr)
    assert m.predict_proba(Xva).shape == (len(Xva), 2)


def test_neural_pipeline_is_seen_as_classifier_by_sklearn_scorers():
    """Regression test: hyper-parameter search scores with PR-AUC (predict_proba), which needs the classifier tag."""
    pytest.importorskip("torch")
    from sklearn.model_selection import cross_val_score
    from churnlab.nn_models import TorchANN
    Xtr, ytr, _, _ = _data()
    s = cross_val_score(_pipe(TorchANN(hidden=(16,), epochs=2)), Xtr.iloc[:600], ytr[:600], scoring="average_precision", cv=2)
