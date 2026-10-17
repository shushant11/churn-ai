"""One tiny function, chat(), that talks to whichever LLM you choose with LLM_PROVIDER:
  none    -> no LLM; the agent answers with templates (default, free, works offline, used by tests/CI)
  mistral -> Mistral API (free tier; set MISTRAL_API_KEY)
  gemini  -> Google Gemini API (free tier; set GEMINI_API_KEY)
  ollama  -> a model running on your own laptop (free; `ollama pull llama3.2`)
Only the Python standard library is used, so there is no SDK to install or version-pin."""
from __future__ import annotations
import json, os, urllib.request
from . import config

MODELS = {"mistral": os.getenv("MISTRAL_MODEL", "mistral-small-latest"), "gemini": os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
          "ollama": os.getenv("OLLAMA_MODEL", "llama3.2")}


class LLMUnavailable(RuntimeError):
    pass


def _post(url, payload, headers, timeout=60):
    req = urllib.request.Request(url, json.dumps(payload).encode(), {"Content-Type": "application/json", **headers})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def chat(messages: list[dict], provider: str | None = None, temperature: float = 0.1) -> str:
    """messages = [{"role": "system"|"user"|"assistant", "content": "..."}]  ->  reply text."""
    provider = (provider or config.LLM_PROVIDER).lower()
    if provider == "none":
        raise LLMUnavailable("LLM_PROVIDER=none")
    try:
        if provider == "mistral":
            key = os.environ["MISTRAL_API_KEY"]
            r = _post("https://api.mistral.ai/v1/chat/completions", {"model": MODELS["mistral"], "messages": messages, "temperature": temperature},
                      {"Authorization": f"Bearer {key}"})
            return r["choices"][0]["message"]["content"]
        if provider == "gemini":
            key = os.environ["GEMINI_API_KEY"]
            sys_txt = "\n".join(m["content"] for m in messages if m["role"] == "system")
            contents = [{"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]} for m in messages if m["role"] != "system"]
            body = {"contents": contents, "generationConfig": {"temperature": temperature}}
            if sys_txt:
                body["systemInstruction"] = {"parts": [{"text": sys_txt}]}
            r = _post(f"https://generativelanguage.googleapis.com/v1beta/models/{MODELS['gemini']}:generateContent?key={key}", body, {})
            return r["candidates"][0]["content"]["parts"][0]["text"]
        if provider == "ollama":
            host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
            r = _post(f"{host}/api/chat", {"model": MODELS["ollama"], "messages": messages, "stream": False, "options": {"temperature": temperature}}, {})
            return r["message"]["content"]
    except KeyError as e:
        raise LLMUnavailable(f"missing environment variable {e}") from e
    except OSError as e:
        raise LLMUnavailable(f"{provider} call failed: {e}") from e
    raise LLMUnavailable(f"unknown provider {provider}")


def available(provider: str | None = None) -> bool:
    try:
        chat([{"role": "user", "content": "Reply with OK"}], provider)
        return True
    except LLMUnavailable:
        return False
