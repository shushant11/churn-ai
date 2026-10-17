"""Dashboard:  streamlit run app/streamlit_app.py
Tabs: Predict (one customer + drivers + what-if) | Portfolio (lift, profit, threshold) | Fairness | Drift | Assistant (agent chat)."""
import json, sys
from pathlib import Path
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from churnlab import agent, config                       # noqa: E402
from churnlab.business import profit_curve                # noqa: E402
from churnlab.monitoring import drift_report, simulate_drift  # noqa: E402
from churnlab.predict import load_bundle                  # noqa: E402

st.set_page_config(page_title="Churn AI", page_icon="📉", layout="wide")


@st.cache_resource
def bundle():
    return load_bundle()


@st.cache_data
def metrics():
    return json.loads((config.REPORTS / "metrics.json").read_text())


@st.cache_data
def holdout():
    return pd.read_csv(config.ARTIFACTS / "test_holdout.csv")


if not config.MODEL_PATH.exists():
    # first launch on a fresh host (the model file is gitignored): train once, quick mode, same seed
    import subprocess, os
    with st.spinner("First launch: training the model (about 1-3 minutes, once)..."):
        res = subprocess.run([sys.executable, "-m", "churnlab.train", "--fast"], capture_output=True, text=True,
                             env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")})
    if res.returncode != 0 or not config.MODEL_PATH.exists():
        st.error("Training failed on first launch."); st.code(res.stderr[-2000:]); st.stop()
    st.cache_resource.clear(); st.cache_data.clear()
b, m = bundle(), metrics()
st.title("Churn AI — risk, reasons and retention")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Model", b.model_name); c2.metric("Test PR-AUC", f"{m['test']['pr_auc']:.3f}", help=f"95% CI {m['test']['pr_auc_95ci']}")
c3.metric("Test ROC-AUC", f"{m['test']['roc_auc']:.3f}"); c4.metric("Campaign threshold", f"{b.threshold:.2f}")

t1, t2, t3, t4, t5 = st.tabs(["Predict", "Portfolio", "Fairness", "Drift", "Assistant"])

with t1:
    with st.form("cust"):
        a, bcol, c = st.columns(3)
        cs = a.number_input("Credit score", 300, 900, 650); geo = a.selectbox("Geography", ["France", "Germany", "Spain"]); gen = a.selectbox("Gender", ["Female", "Male"])
        age = bcol.slider("Age", 18, 92, 42); ten = bcol.slider("Tenure (years)", 0, 10, 5); bal = bcol.number_input("Balance", 0.0, 300000.0, 60000.0, step=1000.0)
        nop = c.selectbox("Number of products", [1, 2, 3, 4]); card = c.checkbox("Has credit card", True); act = c.checkbox("Active member", True)
        sal = c.number_input("Estimated salary", 0.0, 250000.0, 100000.0, step=1000.0)
        go = st.form_submit_button("Predict")
    cust = dict(CreditScore=cs, Geography=geo, Gender=gen, Age=age, Tenure=ten, Balance=bal, NumOfProducts=nop, HasCrCard=int(card), IsActiveMember=int(act), EstimatedSalary=sal)
    if go:
        r = b.predict_one(cust)
        st.subheader(f"Churn probability {r['churn_probability']:.1%} — {r['risk_band']} risk")
        st.progress(min(r["churn_probability"], 1.0))
        st.caption(f"Contact if ≥ {r['threshold']:.2f} (threshold chosen by expected profit on validation data).")
        d = pd.DataFrame(r["drivers"])
        st.bar_chart(d.set_index("feature")["contribution"])
        st.dataframe(d, width="stretch", hide_index=True)
        st.markdown("**What-if:** same customer, but active and with 2 products")
        alt = b.predict_one({**cust, "IsActiveMember": 1, "NumOfProducts": 2}, explain=False)
        st.write(f"{r['churn_probability']:.1%} → {alt['churn_probability']:.1%}  (model estimate, not a causal effect)")

with t2:
    st.markdown("Business assumptions (edit to match your bank):")
    x1, x2, x3 = st.columns(3)
    config.BUSINESS.customer_value = x1.number_input("Value of a retained churner", 0.0, 10000.0, 500.0)
    config.BUSINESS.offer_cost = x2.number_input("Cost per offer", 0.0, 1000.0, 50.0)
    config.BUSINESS.offer_success = x3.slider("Offer success rate", 0.0, 1.0, 0.30)
    h = holdout(); p = b.proba(h)
    curve = profit_curve(h["Exited"].values, p)
    best = curve.loc[curve.profit.idxmax()]
    st.line_chart(curve.set_index("threshold")["profit"])
    st.write(f"Best threshold on the hold-out with these assumptions: **{best.threshold:.2f}**, contacting {int(best.contacted)} customers, profit {best.profit:,.0f}.")
    st.markdown("Decile table (riskiest 10% first)")
    st.dataframe(pd.DataFrame(m["deciles"]), width="stretch", hide_index=True)
    for f in ["gain.png", "calibration.png", "pr_roc.png", "importance.png", "km_geography.png"]:
        if (config.FIGURES / f).exists():
            st.image(str(config.FIGURES / f))

with t3:
    st.markdown("Per-group behaviour at the campaign threshold. A selection-rate ratio below 0.8 triggers a documented review.")
    for attr in ["Gender", "Geography"]:
        st.subheader(attr)
        st.dataframe(pd.DataFrame(m["fairness"][attr]["groups"]), width="stretch", hide_index=True)
        st.json(m["fairness"][attr]["disparity"])
    if "gender_comparison" in m["fairness"]:
        st.subheader("Gender used vs not used (same model, refit)"); st.json(m["fairness"]["gender_comparison"])

with t4:
    ref = pd.read_csv(config.ARTIFACTS / "reference.csv")
    which = st.radio("Compare training data with", ["hold-out (no drift expected)", "simulated next quarter (drift)"])
    cur = holdout() if which.startswith("hold") else simulate_drift(holdout())
    rep = drift_report(ref, cur)
    st.dataframe(rep, width="stretch", hide_index=True)
    st.caption("PSI < 0.1 stable · 0.1–0.25 watch · > 0.25 retrain")

with t5:
    st.caption(f"LLM provider: {config.LLM_PROVIDER} (set LLM_PROVIDER=mistral|gemini|ollama; 'none' uses deterministic rules)")
    st.caption("Try: 'What is the churn risk for customer 15703793?' · 'what if 15703793 products to 2' · 'how is the threshold chosen?'")
    if "chat" not in st.session_state:
        st.session_state.chat = []
    for role, txt in st.session_state.chat:
        st.chat_message(role).write(txt)
    if q := st.chat_input("Ask about a customer or the retention playbook"):
        st.chat_message("user").write(q)
        r = agent.ask(q)
        st.chat_message("assistant").write(r.answer)
        with st.expander("trace: tools, sources, guardrail flags"):
            st.json({"mode": r.mode, "tools": [n for n, _ in r.tool_calls], "flags": r.flags, "sources": [s["section"] for s in r.sources]})
        st.session_state.chat += [("user", q), ("assistant", r.answer)]
