"""The churn-retention assistant: an AGENT = an LLM (or simple rules) that decides which TOOL to call, reads the
result, and answers. Every answer passes the guardrails. Works in two modes:
  rules : no LLM needed. Intent is detected with regex, answers are templates. Deterministic -> easy to test.
  llm   : the LLM plans tool calls in a small JSON loop (ReAct style) and writes the final answer."""
from __future__ import annotations
import json, re
from dataclasses import dataclass, field
from functools import lru_cache
import pandas as pd
from . import config, guardrails, rag
from .llm import LLMUnavailable, chat
from .predict import load_bundle

PROTECTED = {"Gender", "Geography"}


# ------------------------------------------------------------------ tools ----------------------------------------------
@lru_cache(maxsize=1)
def _customers() -> pd.DataFrame:
    return pd.read_csv(config.ARTIFACTS / "test_holdout.csv").set_index("CustomerId")


def lookup_customer(customer_id: int) -> dict:
    df = _customers()
    if int(customer_id) not in df.index:
        return {"error": f"customer {customer_id} not found"}
    row = df.loc[int(customer_id)].drop(labels=[config.TARGET, "RowNumber", "Surname"], errors="ignore")   # never reveal the true label
    return {"customer_id": int(customer_id), **{k: (v.item() if hasattr(v, "item") else v) for k, v in row.items()}}


def predict_churn(customer_id: int) -> dict:
    c = lookup_customer(customer_id)
    if "error" in c:
        return c
    return {"customer_id": int(customer_id), **load_bundle().predict_one({k: v for k, v in c.items() if k != "customer_id"})}


def what_if(customer_id: int, changes: dict) -> dict:
    c = lookup_customer(customer_id)
    if "error" in c:
        return c
    base = {k: v for k, v in c.items() if k != "customer_id"}
    b = load_bundle()
    before = b.predict_one(base, explain=False)["churn_probability"]
    after = b.predict_one({**base, **changes}, explain=False)["churn_probability"]
    return {"customer_id": int(customer_id), "changes": changes, "before": before, "after": after, "delta": round(after - before, 4)}


def search_playbook(query: str, k: int = 3) -> list[dict]:
    return rag.search(query, k)


def top_risk(n: int = 5) -> list[dict]:
    df = _customers()
    b = load_bundle()
    p = b.proba(df.reset_index())
    out = df.assign(p=p).sort_values("p", ascending=False).head(int(n))
    return [{"customer_id": int(i), "churn_probability": round(float(r.p), 4), "NumOfProducts": int(r.NumOfProducts),
             "IsActiveMember": int(r.IsActiveMember), "Age": int(r.Age)} for i, r in out.iterrows()]


TOOLS = {
    "lookup_customer": (lookup_customer, "Get a customer's profile. args: {customer_id:int}"),
    "predict_churn": (predict_churn, "Calibrated churn probability, risk band and top drivers. args: {customer_id:int}"),
    "what_if": (what_if, "Re-score a customer after changing fields, e.g. {IsActiveMember:1}. args: {customer_id:int, changes:object}"),
    "search_playbook": (search_playbook, "Search the bank's retention playbook and policies. args: {query:str}"),
    "top_risk": (top_risk, "Riskiest customers in the hold-out portfolio. args: {n:int}"),
}


def call_tool(name: str, args: dict):
    if name not in TOOLS:
        return {"error": f"unknown tool {name}"}
    try:
        return TOOLS[name][0](**args)
    except Exception as e:                                          # a tool error is shown to the agent, not raised
        return {"error": f"{type(e).__name__}: {e}"}


# ------------------------------------------------------------------ response ---------------------------------------------
@dataclass
class AgentResponse:
    answer: str
    mode: str
    tool_calls: list = field(default_factory=list)
    sources: list = field(default_factory=list)
    flags: list = field(default_factory=list)
    blocked: bool = False


DRIVER_QUERY = {"IsActiveMember": "inactive members re-engagement", "Age": "older customers retirement", "Balance": "high balance customers rate",
                "Tenure": "single product customers early relationship", "CreditScore": "offer budget", "EstimatedSalary": "offer budget",
                "HasCrCard": "single product customers second product"}


def _driver_query(d: dict) -> str:
    if d["feature"] == "NumOfProducts":
        return "three or more products fatigue" if d["value"] >= 3 else "single product customers second product"
    return DRIVER_QUERY.get(d["feature"], "risk bands")


def _cite(hit: dict) -> str:
    return f"[source: {hit['source']} > {hit['section'].split(' > ')[-1]}]"


# ------------------------------------------------------------------ rules mode -------------------------------------------
CHANGE_PATTERNS = [(r"\b(became|become|becomes|was|is|were) active|activate", {"IsActiveMember": 1}),
                   (r"products? (to|=) ?(\d)", None), (r"balance (to|=) ?([\d,\.]+)", None), (r"\b(credit ?card)\b", {"HasCrCard": 1})]


def _parse_changes(q: str) -> dict:
    ch = {}
    low = q.lower()
    if re.search(CHANGE_PATTERNS[0][0], low):
        ch["IsActiveMember"] = 1
    m = re.search(r"products? (?:to|=) ?(\d)", low) or re.search(r"(\d) products?", low)
    if m:
        ch["NumOfProducts"] = int(m.group(1))
    m = re.search(r"balance (?:to|=|of) ?([\d,\.]+)", low)
    if m:
        ch["Balance"] = float(m.group(1).replace(",", ""))
    if re.search(r"\b(gets?|got|adds?|with) (a )?credit ?card", low):
        ch["HasCrCard"] = 1
    return ch


def _rules(q: str) -> AgentResponse:
    calls, low = [], q.lower()
    ids = re.findall(r"\b(1\d{7})\b", q)
    if ids and re.search(r"what if|what happens if|if (he|she|they|the customer)|would .* change", low):
        ch = _parse_changes(q)
        if not ch:
            return AgentResponse("Tell me what to change, e.g. 'what if 15634602 became active' or 'products to 2'.", "rules")
        r = call_tool("what_if", {"customer_id": int(ids[0]), "changes": ch}); calls.append(("what_if", r))
        if "error" in r:
            return AgentResponse(r["error"], "rules", calls)
        same = {k: v for k, v in ch.items() if lookup_customer(int(ids[0])).get(k) == v}
        if same and len(same) == len(ch):
            return AgentResponse(f"Customer {ids[0]} already has {', '.join(f'{k}={v}' for k, v in same.items())}, so nothing changes. "
                                 "Try a different change, e.g. products to 2.", "rules", calls)
        direction = "lowers" if r["delta"] < 0 else "raises"
        return AgentResponse(f"For customer {r['customer_id']}, changing {', '.join(f'{k} to {v}' for k, v in ch.items())} "
                             f"{direction} the churn probability from {r['before']} to {r['after']} (change {r['delta']:+}). "
                             "This is what the model predicts, not a proven causal effect.", "rules", calls)
    if ids:
        r = call_tool("predict_churn", {"customer_id": int(ids[0])}); calls.append(("predict_churn", r))
        if "error" in r:
            return AgentResponse(r["error"], "rules", calls)
        drivers = [d for d in r["drivers"] if d["contribution"] > 0]
        behavioural = [d for d in drivers if d["feature"] not in PROTECTED]
        txt = (f"Customer {r['customer_id']} has a churn probability of {r['churn_probability']} ({r['risk_band']} risk; campaign threshold {r['threshold']}). ")
        if drivers:
            txt += "Drivers pushing risk up: " + "; ".join(f"{d['feature']}={d['value']} ({d['contribution']:+})" for d in drivers[:4]) + ". "
        if any(d["feature"] in PROTECTED for d in drivers):
            txt += "Note: Gender and Geography are listed for transparency only; actions are chosen from behavioural drivers. "
        sources = []
        if r["risk_band"] == "low":
            txt += "No retention action is needed for a low-risk customer."
        elif behavioural:
            hit = (call_tool("search_playbook", {"query": _driver_query(behavioural[0]), "k": 1}) or [None])[0]
            calls.append(("search_playbook", [hit]))
            if hit:
                sources.append(hit)
                first = re.split(r"(?<=\.) ", hit["text"])
                txt += "Recommended action: " + " ".join(first[:3]) + " " + _cite(hit)
        return AgentResponse(txt.strip(), "rules", calls, sources)
    if re.search(r"\b(top|riskiest|highest risk|most at risk|portfolio)\b", low):
        n = int((re.search(r"\b(\d{1,2})\b", low) or [None, 5])[1])
        r = call_tool("top_risk", {"n": n}); calls.append(("top_risk", r))
        if isinstance(r, dict) and "error" in r:                       # a tool failure is reported, never crashes the assistant
            return AgentResponse("I couldn't read the portfolio: " + r["error"], "rules", calls)
        return AgentResponse("Highest-risk customers in the hold-out portfolio: " + "; ".join(f"{c['customer_id']} ({c['churn_probability']})" for c in r) + ".", "rules", calls)
    hits = call_tool("search_playbook", {"query": q, "k": 2}); calls.append(("search_playbook", hits))
    if not hits or hits[0]["score"] < 0.08:
        return AgentResponse("I couldn't find this in the retention playbook or policies, so I won't guess.", "rules", calls)
    h = hits[0]
    return AgentResponse(" ".join(re.split(r"(?<=\.) ", h["text"])[:3]) + " " + _cite(h), "rules", calls, [h])


# ------------------------------------------------------------------ LLM mode ---------------------------------------------
SYSTEM = """You are a bank's churn-retention assistant. You MUST use tools for every fact; never invent numbers.
Tools:
{tools}
Reply with ONE JSON object and nothing else, either
  {{"tool": "<name>", "args": {{...}}}}            to call a tool, or
  {{"answer": "<final answer>"}}                    when done.
Rules: quote probabilities exactly as tools return them. When you use the playbook, cite it as [source: file > section].
Never give gender or nationality as the reason for an action. If the tools don't contain the answer, say you don't know."""


def _json(text: str):
    m = re.search(r"\{.*\}", text, re.S)
    return json.loads(m.group(0)) if m else None


def _llm(q: str, max_steps: int = 5) -> AgentResponse:
    msgs = [{"role": "system", "content": SYSTEM.format(tools="\n".join(f"- {k}: {v[1]}" for k, v in TOOLS.items()))},
            {"role": "user", "content": q}]
    calls, sources = [], []
    for _ in range(max_steps):
        out = chat(msgs)
        try:
            obj = _json(out)
        except json.JSONDecodeError:
            obj = None
        if not obj:
            msgs += [{"role": "assistant", "content": out}, {"role": "user", "content": "Reply with valid JSON only."}]
            continue
        if "answer" in obj:
            return AgentResponse(obj["answer"], "llm", calls, sources)
        res = call_tool(obj.get("tool", ""), obj.get("args", {}))
        calls.append((obj.get("tool"), res))
        if obj.get("tool") == "search_playbook" and isinstance(res, list):
            sources += res
        msgs += [{"role": "assistant", "content": json.dumps(obj)}, {"role": "user", "content": "TOOL RESULT: " + json.dumps(res, default=str)[:4000]}]
    return AgentResponse("I couldn't finish within the step limit.", "llm", calls, sources)


# ------------------------------------------------------------------ entry point -------------------------------------------
def ask(question: str, use_llm: bool | None = None) -> AgentResponse:
    g = guardrails.check_input(question)
    if not g.allowed:
        return AgentResponse(guardrails.REFUSAL.get(g.flags[-1], "Request blocked."), "guardrail", flags=g.flags, blocked=True)
    q = g.text
    use_llm = config.LLM_PROVIDER != "none" if use_llm is None else use_llm
    resp = None
    if use_llm:
        try:
            resp = _llm(q)
        except LLMUnavailable as e:
            resp = None; g.flags.append(f"llm_unavailable:{e}")
    if resp is None:
        resp = _rules(q)
    tool_txt = [json.dumps(r, default=str) for _, r in resp.tool_calls]
    out = guardrails.check_output(resp.answer, tool_txt + [q], needs_citation=bool(resp.sources))
    resp.flags = g.flags + out.flags
    if not out.allowed and resp.mode == "llm":                      # LLM broke a rule -> fall back to the safe deterministic answer
        safe = _rules(q); safe.flags = resp.flags + ["fell_back_to_rules"]
        return safe
    return resp
