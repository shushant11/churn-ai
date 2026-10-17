import json
from churnlab.predict import load_bundle

CUSTOMER = dict(CreditScore=600, Geography="Germany", Gender="Female", Age=50, Tenure=3, Balance=120000.0,
                NumOfProducts=3, HasCrCard=1, IsActiveMember=0, EstimatedSalary=90000.0)


def test_prediction_is_json_serialisable():
    """FastAPI/Streamlit/MCP need plain Python types; numpy ints used to break the API response."""
    r = load_bundle().predict_one(CUSTOMER)
    json.dumps(r)                       # raises TypeError if any numpy scalar is left
    assert 0 <= r["churn_probability"] <= 1 and len(r["drivers"]) == 5
