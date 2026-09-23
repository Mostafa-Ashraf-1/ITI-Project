"""
Generation service: builds a grounded prompt from retrieved context and calls Groq.
Falls back to a deterministic, non-hallucinating extractive answer when Groq is
unavailable, so the API does not fabricate a legal answer it cannot generate.
"""
from groq import Groq

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


def _call_groq(prompt: str) -> str:
    if not settings.GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not configured")

    client = Groq(api_key=settings.GROQ_API_KEY)

    response = client.chat.completions.create(
        model=settings.GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        stream=False,
        temperature=0.1,
        max_completion_tokens=1024,
    )

    answer = response.choices[0].message.content
    if not answer:
        raise RuntimeError("Groq returned an empty response")

    return answer.strip()


def _extractive_fallback(contexts: list) -> str:
    if not contexts:
        return "The retrieved context does not contain this information."

    lead = contexts[0]
    body = lead["text"].strip().replace("\n", " ")
    if len(body) > 600:
        body = body[:600] + "..."

    cite = (
        f"(Article {lead['article']}, p. {lead['page']})"
        if lead.get("article")
        else f"(p. {lead.get('page')})"
    )
    return f"Based on the retrieved text {cite}: {body}"


def generate_answer(question: str, contexts: list):
    """Return (answer_text, backend_used)."""
    prompt = _build_prompt(question, contexts)

    try:
        answer = _call_groq(prompt)
        return answer, f"groq:{settings.GROQ_MODEL}"
    except Exception:
        return _extractive_fallback(contexts), "extractive-fallback (Groq unavailable)"
