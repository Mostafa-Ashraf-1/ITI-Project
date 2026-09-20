"""
Generation service: builds a grounded prompt from retrieved context and calls a
local Ollama server. Falls back to a deterministic, non-hallucinating extractive
answer when no Ollama server is reachable, so the API never fabricates a legal
answer it can't actually generate.
"""
import json
import urllib.request

from app.core.config import settings

SYSTEM_PROMPT = (
    "You are a legal assistant answering questions about the Egyptian Civil Code. "
    "Answer strictly and only using the CONTEXT provided below, which was retrieved "
    "from the actual text of the Egyptian Civil Code. Do not use outside knowledge and "
    "do not invent article numbers, facts, or legal rules. If the answer is not present "
    "in the context, say explicitly: 'The retrieved context does not contain this "
    "information.' Always cite the Article number(s) and page(s) you relied on."
)


def _build_prompt(question: str, contexts: list) -> str:
    ctx_block = "\n\n".join(
        f"[Article {c.get('article')}, page {c.get('page')}] {c['text']}" for c in contexts
    )
    return f"{SYSTEM_PROMPT}\n\nCONTEXT:\n{ctx_block}\n\nQUESTION: {question}\n\nANSWER:"


def _call_ollama(prompt: str, timeout: int = 30) -> str:
    req = urllib.request.Request(
        f"{settings.OLLAMA_HOST}/api/generate",
        data=json.dumps({"model": settings.OLLAMA_MODEL, "prompt": prompt, "stream": False}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["response"]


def _extractive_fallback(contexts: list) -> str:
    if not contexts:
        return "The retrieved context does not contain this information."
    lead = contexts[0]
    body = lead["text"].strip().replace("\n", " ")
    if len(body) > 600:
        body = body[:600] + "..."
    cite = f"(Article {lead['article']}, p. {lead['page']})" if lead.get("article") else f"(p. {lead.get('page')})"
    return f"Based on the retrieved text {cite}: {body}"


def generate_answer(question: str, contexts: list):
    """Returns (answer_text, backend_used)."""
    prompt = _build_prompt(question, contexts)
    try:
        answer = _call_ollama(prompt)
        return answer, f"ollama:{settings.OLLAMA_MODEL}"
    except Exception:
        return _extractive_fallback(contexts), "extractive-fallback (no ollama server reachable)"
