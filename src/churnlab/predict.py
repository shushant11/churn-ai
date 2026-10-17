"""The ONE object the API, dashboard, agent and MCP server all load. Training and serving share it, so they cannot disagree."""
from __future__ import annotations
from dataclasses import dataclass, field
import joblib
import numpy as np
import pandas as pd
from . import config
from .explain import local_drivers
from .features import RAW


@dataclass
class ModelBundle:
    pipeline: object                 # features + estimator (sklearn Pipeline)
    calibrator: object               # PlattCalibrator fitted on validation
    threshold: float                 # profit-optimal cut-off chosen on validation
    model_name: str
    background: pd.DataFrame         # small sample of TRAIN rows (for explanations and drift reference)
    metadata: dict = field(default_factory=dict)

    def proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.calibrator.transform(self.pipeline.predict_proba(X[RAW])[:, 1])

    def risk_band(self, p: float) -> str:
        return "high" if p >= max(self.threshold, 0.5) else "medium" if p >= self.threshold * 0.6 else "low"

    def predict_one(self, customer: dict, explain: bool = True) -> dict:
        row = pd.DataFrame([customer])[RAW]
        p = float(self.proba(row)[0])
        out = {"churn_probability": round(p, 4), "will_churn": bool(p >= self.threshold), "threshold": round(self.threshold, 4),
               "risk_band": self.risk_band(p), "model": self.model_name}
        if explain:
            out["drivers"] = local_drivers(self.proba, row, self.background,
                                           skip=() if self.metadata.get("uses_gender", True) else ("Gender",))
        return out

    def save(self, path=config.MODEL_PATH):
        joblib.dump(self, path)
        return path


_CACHE: dict = {}


def load_bundle(path=config.MODEL_PATH) -> ModelBundle:
    """Loaded once per process and cached (the original app re-read the model on every click)."""
    key = str(path)
    if key not in _CACHE:
        try:
            _CACHE[key] = joblib.load(path)
        except Exception as e:                  # usually: saved with different library versions than the installed ones
            raise RuntimeError(f"Could not load {path} ({type(e).__name__}). It was probably trained with other library versions. "
                               "Fix: delete it and run  python -m churnlab.train") from e
    return _CACHE[key]
