"""The deployed model is gender-blind: flipping Gender must not change a single prediction."""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from churnlab.data import load, split, xy
from churnlab.features import add_features, make_preprocessor
from churnlab.models import Blend, _pipe, with_gender


def _data():
    tr, va, _ = split(load())
    return (*xy(tr.head(1500)), *xy(va.head(400)))


def test_blind_preprocessor_has_no_gender_columns():
    Xtr, _, _, _ = _data()
    assert "Male" not in add_features(Xtr, drop_gender=True).columns
    out = make_preprocessor(drop_gender=True).fit_transform(Xtr)
    assert not any("Male" in c or "Gender" in c for c in out.columns)
    assert "Male" in make_preprocessor(drop_gender=False).fit_transform(Xtr).columns


def test_flipping_gender_does_not_change_predictions():
    Xtr, ytr, Xva, _ = _data()
    m = _pipe(RandomForestClassifier(n_estimators=40, random_state=0), scale=False).fit(Xtr, ytr)    # blind by default
    flipped = Xva.assign(Gender=Xva["Gender"].map({"Male": "Female", "Female": "Male"}))
    assert np.allclose(m.predict_proba(Xva), m.predict_proba(flipped))


def test_with_gender_builds_the_opposite_model():
    Xtr, ytr, Xva, _ = _data()
    base = _pipe(RandomForestClassifier(n_estimators=40, random_state=0), scale=False).fit(Xtr, ytr)
    aware = with_gender(base, use_gender=True).fit(Xtr, ytr)
    assert "Male" in aware.named_steps["prep"].transform(Xva.head(5)).columns
    blend = with_gender(Blend.from_fitted(["a"], [base]), use_gender=True).fit(Xtr, ytr)
    assert blend.predict_proba(Xva).shape == (len(Xva), 2)
