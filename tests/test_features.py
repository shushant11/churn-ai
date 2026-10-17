import pandas as pd
from churnlab.data import load, xy
from churnlab.features import add_features, make_preprocessor


def test_engineered_features():
    X, _ = xy(load().head(200))
    f = add_features(X)
    assert {"BalanceSalaryRatio", "ZeroBalance", "MultiProduct", "Male"} <= set(f.columns)
    assert ((f.NumOfProducts >= 3) == (f.MultiProduct == 1)).all()


def test_preprocessor_shape_and_unknown_category():
    X, _ = xy(load().head(300))
    pre = make_preprocessor().fit(X)
    out = pre.transform(X)
    assert out.shape[0] == 300 and not out.isna().any().any()
    weird = X.head(1).copy(); weird["Geography"] = "Italy"          # unseen category must not crash serving
    assert pre.transform(weird).shape[1] == out.shape[1]
