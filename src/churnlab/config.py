"""Single place for paths, seeds and business assumptions."""
from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = Path(os.getenv("CHURN_DATA", ROOT / "data" / "Churn_Modelling.csv"))
ARTIFACTS = ROOT / "artifacts"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
KNOWLEDGE = ROOT / "knowledge"
for p in (ARTIFACTS, REPORTS, FIGURES):
    p.mkdir(parents=True, exist_ok=True)

SEED = 42
TARGET = "Exited"
DROP_COLS = ["RowNumber", "CustomerId", "Surname"]          # identifiers: no predictive meaning, would leak identity
MODEL_PATH = ARTIFACTS / "model.joblib"
# Gender-blind by default: the model never sees Gender. Set CHURN_USE_GENDER=1 (or train with --keep-gender) to compare.
DROP_GENDER = os.getenv("CHURN_USE_GENDER", "0") != "1"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "none").lower()    # none | mistral | gemini | ollama


@dataclass
class Business:
    """ASSUMPTIONS - change them to match a real bank. The data contains no money figures for retention."""
    customer_value: float = 500.0     # value kept if a churner is retained (currency units)
    offer_cost: float = 50.0          # cost of one retention offer, paid for every customer contacted
    offer_success: float = 0.30       # share of true churners who accept the offer and stay

    def profit(self, tp: int, fp: int) -> float:
        return tp * (self.offer_success * self.customer_value - self.offer_cost) - fp * self.offer_cost


BUSINESS = Business()
