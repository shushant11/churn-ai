"""REST API:  uvicorn churnlab.api:app --reload   ->  http://127.0.0.1:8000/docs  (interactive Swagger page)
pydantic validates every request, so a negative age or 'Germny' is rejected with a clear 422 error instead of a silent bad prediction."""
from __future__ import annotations
from contextlib import asynccontextmanager
from typing import Literal
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from . import __version__, agent, config
from .monitoring import drift_report
from .predict import load_bundle


class Customer(BaseModel):
    CreditScore: int = Field(..., ge=300, le=900, examples=[650])
    Geography: Literal["France", "Germany", "Spain"]
    Gender: Literal["Female", "Male"]
    Age: int = Field(..., ge=18, le=100, examples=[42])
    Tenure: int = Field(..., ge=0, le=10)
    Balance: float = Field(..., ge=0)
    NumOfProducts: int = Field(..., ge=1, le=4)
    HasCrCard: Literal[0, 1]
    IsActiveMember: Literal[0, 1]
    EstimatedSalary: float = Field(..., ge=0)


class Question(BaseModel):
    question: str = Field(..., min_length=2, max_length=2000)


@asynccontextmanager
async def lifespan(_app):
    if config.MODEL_PATH.exists():
        load_bundle()                      # load once at start-up, not on every request
    yield


app = FastAPI(title="Churn AI", version=__version__, lifespan=lifespan,
              description="Calibrated churn risk, explanations and a guard-railed retention assistant.")


@app.get("/health")
def health():
    ok = config.MODEL_PATH.exists()
    meta = load_bundle().metadata if ok else {}
    return {"status": "ok" if ok else "model_missing", "version": __version__, "model": load_bundle().model_name if ok else None, **meta}


@app.post("/predict")
def predict(c: Customer, explain: bool = True):
    if not config.MODEL_PATH.exists():
        raise HTTPException(503, "model not trained - run `python -m churnlab.train`")
    return load_bundle().predict_one(c.model_dump(), explain=explain)


@app.post("/predict/batch")
def predict_batch(customers: list[Customer]):
    if len(customers) > 5000:
        raise HTTPException(413, "max 5000 rows per call")
    b = load_bundle()
    p = b.proba(pd.DataFrame([c.model_dump() for c in customers]))
    return [{"churn_probability": round(float(x), 4), "will_churn": bool(x >= b.threshold), "risk_band": b.risk_band(float(x))} for x in p]


@app.post("/drift")
def drift(customers: list[Customer]):
    ref = pd.read_csv(config.ARTIFACTS / "reference.csv")
    return drift_report(ref, pd.DataFrame([c.model_dump() for c in customers])).to_dict(orient="records")


@app.post("/assistant")
def assistant(q: Question):
    r = agent.ask(q.question)
    return {"answer": r.answer, "mode": r.mode, "blocked": r.blocked, "flags": r.flags, "tools": [n for n, _ in r.tool_calls],
            "sources": [{"source": s["source"], "section": s["section"]} for s in r.sources]}
