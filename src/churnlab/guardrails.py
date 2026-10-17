"""Guardrails = checks around the LLM, like a bank's 'four-eyes' rule.
INPUT  : block prompt-injection, redact personal data (PII), keep the assistant on topic, cap length.
OUTPUT : every number must come from a tool result (no invented probabilities), no protected attribute
         (gender, nationality) may be given as the REASON for an action, and playbook claims must cite a source."""
from __future__ import annotations
import re
from dataclasses import dataclass, field

MAX_CHARS = 2000
INJECTION = [r"ignore (all |any )?(previous|prior|above) (instructions|rules)", r"disregard (the )?(system|previous)", r"you are now",
             r"reveal (your|the) (system )?prompt", r"\bjailbreak\b", r"act as (an? )?(unfiltered|dan)", r"developer mode",
             r"print (your|the) (instructions|prompt)", r"override (the )?(guardrails|rules|policy)"]
PII = {"EMAIL": r"[\w.+-]+@[\w-]+\.[\w.]+", "CARD": r"\b(?:\d[ -]?){13,16}\b", "PHONE": r"(?<!\d)(?:\+?\d{1,3}[ -]?)?\d{10}(?!\d)",
       "IBAN": r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b"}
ON_TOPIC = ["churn", "customer", "retention", "risk", "offer", "predict", "probability", "why", "explain", "segment", "bank", "balance",
            "product", "tenure", "playbook", "policy", "model", "drift", "fair", "campaign", "what if", "credit", "active", "germany",
            "france", "spain", "age", "leave", "stay", "threshold", "score", "id", "account", "member", "salary", "help", "hello", "hi"]
PROTECTED_REASON = r"\b(because|since|as|due to)\b[^.]{0,40}\b(she|he|female|male|woman|man|gender|german|french|spanish|nationality)\b"


@dataclass
class GuardResult:
    allowed: bool
    text: str
    flags: list = field(default_factory=list)


def check_input(text: str) -> GuardResult:
    flags = []
    if len(text) > MAX_CHARS:
        return GuardResult(False, "", ["too_long"])
    low = text.lower()
    if any(re.search(p, low) for p in INJECTION):
        return GuardResult(False, "", ["prompt_injection"])
    clean = text
    for tag, pat in PII.items():
        if re.search(pat, clean):
            clean = re.sub(pat, f"[{tag}]", clean); flags.append(f"pii_{tag.lower()}")
    if not re.search(r"\b(" + "|".join(ON_TOPIC) + ")", low) and not re.search(r"\b\d{8}\b", low):
        return GuardResult(False, clean, flags + ["off_topic"])
    return GuardResult(True, clean, flags)


def _numbers(s: str) -> set[str]:
    return {n.rstrip(".") for n in re.findall(r"\d+(?:\.\d+)?%?", s)}


def check_output(answer: str, tool_outputs: list[str], needs_citation: bool = False) -> GuardResult:
    flags = []
    grounded = " ".join(tool_outputs)
    allowed_nums = _numbers(grounded)
    # also allow a probability written as a percentage (0.734 -> 73.4%) and small counting numbers
    for n in list(allowed_nums):
        try:
            v = float(n.rstrip("%"))
            if v <= 1:
                allowed_nums |= {f"{v*100:.0f}%", f"{v*100:.1f}%", f"{v*100:.0f}", f"{v*100:.1f}"}
        except ValueError:
            pass
    invented = [n for n in _numbers(answer) if n not in allowed_nums and not (n.isdigit() and int(n) <= 10)]
    if invented:
        flags.append(f"ungrounded_numbers:{','.join(sorted(invented)[:5])}")
    if re.search(PROTECTED_REASON, answer.lower()):
        flags.append("protected_attribute_as_reason")
    if needs_citation and "[source:" not in answer.lower():
        flags.append("missing_citation")
    return GuardResult(not flags, answer, flags)


REFUSAL = {"prompt_injection": "I can't follow instructions that try to change my rules. I can help with churn risk, drivers and retention actions.",
           "off_topic": "I'm the churn-retention assistant, so I can only help with customer churn, risk drivers and retention playbooks.",
           "too_long": f"That message is too long (limit {MAX_CHARS} characters). Please shorten it."}
