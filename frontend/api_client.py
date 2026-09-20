import os
import requests

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")


def check_health(timeout=5):
    resp = requests.get(f"{BACKEND_URL}/health", timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def ask_question(question: str, top_k: int = 3, timeout=60):
    resp = requests.post(
        f"{BACKEND_URL}/query",
        json={"question": question, "top_k": top_k},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()
