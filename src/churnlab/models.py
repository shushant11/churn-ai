"""Model zoo. Everything is a full Pipeline (features + estimator) with the same fit / predict_proba interface."""
from __future__ import annotations
import warnings
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from . import config
from .features import make_preprocessor


def _pipe(clf, scale=True):
    return Pipeline([("prep", make_preprocessor(scale)), ("clf", clf)])


def candidates() -> dict:
    s = config.SEED
    m = {
        "logreg": _pipe(LogisticRegression(max_iter=2000, class_weight="balanced")),
        "random_forest": _pipe(RandomForestClassifier(n_estimators=400, min_samples_leaf=5, class_weight="balanced_subsample", n_jobs=-1, random_state=s), scale=False),
        "hist_gb": _pipe(HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300, class_weight="balanced", random_state=s), scale=False),
        "mlp_sklearn": _pipe(MLPClassifier((64, 32), early_stopping=True, validation_fraction=0.15, n_iter_no_change=10, random_state=s, max_iter=300)),
    }
    try:
        from xgboost import XGBClassifier
        m["xgboost"] = _pipe(XGBClassifier(n_estimators=400, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", random_state=s, n_jobs=-1), scale=False)
    except ImportError:
        warnings.warn("xgboost not installed - skipped")
    try:
        from lightgbm import LGBMClassifier
        m["lightgbm"] = _pipe(LGBMClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, class_weight="balanced", random_state=s, verbose=-1), scale=False)
    except ImportError:
        warnings.warn("lightgbm not installed - skipped")
    try:
        from .nn_models import TorchANN
        m["ann_pytorch"] = _pipe(TorchANN(random_state=s))
    except ImportError:
        warnings.warn("torch not installed - PyTorch ANN skipped")
    try:
        from .nn_models import TorchFT
        m["ft_transformer"] = _pipe(TorchFT(random_state=s))
    except ImportError:
        pass                                    # torch warning already given above
    try:
        from .nn_models import KerasANN
        m["ann_keras"] = _pipe(KerasANN(random_state=s))
    except ImportError:
        warnings.warn("tensorflow not installed - Keras ANN skipped")
    return m


def tuned_hist_gb(X, y, n_trials: int = 25):
    """Hyper-parameter search on TRAIN only (stratified 5-fold CV, PR-AUC). Optuna if installed, else randomised search."""
    from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
    base = _pipe(HistGradientBoostingClassifier(class_weight="balanced", random_state=config.SEED), scale=False)
    space = {"clf__max_depth": [2, 3, 4, 5, None], "clf__learning_rate": [0.02, 0.03, 0.05, 0.08, 0.1], "clf__max_iter": [150, 250, 400],
             "clf__min_samples_leaf": [10, 20, 40, 80], "clf__l2_regularization": [0.0, 0.1, 1.0, 5.0]}
    rs = RandomizedSearchCV(base, space, n_iter=n_trials, scoring="average_precision", cv=StratifiedKFold(5, shuffle=True, random_state=config.SEED),
                            random_state=config.SEED, n_jobs=-1, refit=True)
    rs.fit(X, y)
    return rs.best_estimator_, {"best_params": {k: (v if v is None or isinstance(v, (int, float, str)) else str(v)) for k, v in rs.best_params_.items()}, "cv_pr_auc": float(rs.best_score_)}


def _json_safe(params: dict) -> dict:
    return {k: (v if v is None or isinstance(v, (int, float, str)) else str(v)) for k, v in params.items()}


def tune_pipeline(base, space: dict, X, y, n_trials: int = 10, cv: int = 3):
    """Random search on TRAIN ONLY (stratified CV, PR-AUC). Validation and test sets are never used here."""
    from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
    rs = RandomizedSearchCV(base, space, n_iter=n_trials, scoring="average_precision", cv=StratifiedKFold(cv, shuffle=True, random_state=config.SEED),
                            random_state=config.SEED, n_jobs=1, refit=True, error_score="raise")
    rs.fit(X, y)
    return rs.best_estimator_, {"best_params": _json_safe(rs.best_params_), "cv_pr_auc": float(rs.best_score_), "trials": n_trials}


def tuned_networks(X, y, n_trials: int = 10) -> dict:
    """Hyper-parameter search for every neural model that is installed. Returns {name: (fitted_estimator, info)}."""
    out, s = {}, config.SEED
    try:
        from .nn_models import TorchANN
        out["ann_pytorch_tuned"] = tune_pipeline(_pipe(TorchANN(random_state=s)), {
            "clf__hidden": [(32,), (64,), (64, 32), (128, 64), (128, 64, 32)], "clf__lr": [3e-4, 1e-3, 3e-3],
            "clf__dropout": [0.0, 0.1, 0.2, 0.3, 0.5], "clf__weight_decay": [0.0, 1e-5, 1e-4, 1e-3], "clf__batch": [64, 128, 256]}, X, y, n_trials)
    except ImportError:
        pass
    try:
        from .nn_models import TorchFT
        out["ft_transformer_tuned"] = tune_pipeline(_pipe(TorchFT(random_state=s)), {
            "clf__d_model": [16, 32, 64], "clf__n_layers": [1, 2, 3], "clf__dropout": [0.0, 0.1, 0.2, 0.3],
            "clf__lr": [3e-4, 1e-3, 3e-3], "clf__weight_decay": [1e-5, 1e-4, 1e-3]}, X, y, n_trials)
    except ImportError:
        pass
    try:
        from .nn_models import KerasANN
        out["ann_keras_tuned"] = tune_pipeline(_pipe(KerasANN(random_state=s)), {
            "clf__hidden": [(32,), (64,), (64, 32), (128, 64)], "clf__lr": [3e-4, 1e-3, 3e-3], "clf__dropout": [0.0, 0.1, 0.2, 0.3]}, X, y, max(4, n_trials // 2))
    except ImportError:
        pass
    return out


class Blend(ClassifierMixin, BaseEstimator):
    """Average the churn probabilities of several full pipelines. The members and the blend are judged on the VALIDATION set only."""
    def __init__(self, members=(), names=()):
        self.members, self.names = members, names

    @classmethod
    def from_fitted(cls, names, fitted):
        b = cls(members=list(fitted), names=tuple(names))
        b.fitted_, b.classes_ = list(fitted), np.array([0, 1])      # reuse models that are already trained
        return b

    def fit(self, X, y):                      # used by the gender-blind ablation (clone -> refit every member)
        self.fitted_ = [clone(m).fit(X, y) for m in self.members]
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        p = np.mean([m.predict_proba(X)[:, 1] for m in self.fitted_], axis=0)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def with_gender(model, use_gender: bool):
    """Unfitted copy of `model` (a pipeline or a Blend) whose feature step does / does not see Gender - for the fairness comparison."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from .features import make_preprocessor
    m = clone(model)
    pipes = list(m.members) if isinstance(m, Blend) else [m]
    for p in pipes:
        assert isinstance(p, Pipeline) and "prep" in p.named_steps
        scale = isinstance(p.named_steps["prep"].named_steps["encode"].transformers[0][1], StandardScaler)
        p.set_params(prep=make_preprocessor(scale, drop_gender=not use_gender))
    return m
