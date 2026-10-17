import pandas as pd
from churnlab import config
from churnlab.data import DataValidationError, load, split, validate


def test_load_and_validate():
    df = load()
    info = validate(df)
    assert info["rows"] == 10000 and 0.19 < info["churn_rate"] < 0.22


def test_validation_catches_bad_rows():
    df = load().head(50).copy()
    df.loc[0, "Age"] = -5
    df.loc[1, "Geography"] = "Germny"
    try:
        validate(df)
        assert False, "should have raised"
    except DataValidationError as e:
        assert "Age" in str(e) and "Geography" in str(e)


def test_split_is_stratified_and_disjoint():
    tr, va, te = split(load())
    assert (len(tr), len(va), len(te)) == (6000, 2000, 2000)
    for d in (tr, va, te):
        assert abs(d[config.TARGET].mean() - 0.2037) < 0.01
    assert not set(tr.CustomerId) & set(te.CustomerId)
    assert not set(va.CustomerId) & set(te.CustomerId)
